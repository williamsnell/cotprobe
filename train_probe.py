import wandb
import re
import tqdm
import torch as t
from torch.utils.data import Dataset, DataLoader
import polars as rs
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

device = "cuda" if t.cuda.is_available() else "cpu"



# Define a custom dataset class for the Polars DataFrame
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        # Trim off the last character (the answer) so the LLM regenerates it
        self.texts = [text[:-1] for text in dataframe["text"].to_list()]
        self.labels = dataframe["numeric label"].to_list()
        self.tokenizer = tokenizer
        self.max_length = max([len(text) for text in self.texts])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text (you might need to adjust this based on your tokenizer)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Remove batch dimension added by tokenizer
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": t.tensor(label, dtype=t.long)
        }


class ProbeQwen(t.nn.Module):
    def __init__(self, base_model_name, out_features=10, load_in_8bit=False):
        super().__init__()
        # In qwen, .model is everything up until the unembed.
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_8bit=load_in_8bit)
        self.base_model = base_model.model
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Simple linear probe for now
        in_features = base_model.lm_head.in_features
        self.probe = t.nn.Linear(in_features, out_features, bias=False)

        self.to(device)

    def forward(self, *args, **kwargs):
        with t.no_grad():
            residual_stream = self.base_model(*args, output_scores=True, **kwargs).last_hidden_state

        return self.probe(residual_stream.detach())
        

    def from_text(self, text: str):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True).to(device)
        return self.forward(tokens['input_ids'])        


class Trainer(t.nn.Module):
    def __init__(self, model_name, learning_rate=2e-5):
        super().__init__()
        self.model = ProbeQwen(model_name)
        
        self.optimizer = t.optim.AdamW(self.model.probe.parameters(), lr=learning_rate)

        self.loss_criterion = t.nn.CrossEntropyLoss()

        wandb.watch(self.model.probe, self.loss_criterion, log="all", log_freq=10)


    def train(self, epochs=5, positions_to_predict=None):

        loss_criterion = self.loss_criterion 

        for epoch in tqdm.tqdm(range(epochs)):
            self.model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).unsqueeze(-1).repeat(1, input_ids.shape[-1])

                self.optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # we should be able to always predict from the ~last token,
                # if the probe is learning anything, since that's just a 
                # constrained unembed

                # Allow only predicting the last n tokens
                if positions_to_predict is not None:
                    logits = logits[:, -positions_to_predict:]
                    labels = labels[:, -positions_to_predict:]

                loss = loss_criterion(logits.flatten(0, 1), labels.flatten(0, 1))

                tqdm.tqdm.write(f"Loss: {loss.item()}")
                    


                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                print(labels[:, -1])
                print(logits[:, -1].softmax(dim=-1))
                print(labels[:, -1])

                # Log batch-level metrics (every 10 batches)
                if batch_idx % 2 == 0:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch": batch_idx,
                        "train/global_step": epoch * len(train_loader) + batch_idx,
                        "train/prob correct last token": logits[:, -1].softmax(dim=-1)[range(logits.shape[0]), labels[:, -1].int()].mean(),
                        "train/prob correct all tokens": logits.softmax(dim=-1).flatten(0, 1)[range(logits.shape[0] * logits.shape[1]), labels.flatten(0, 1)].mean(),
                    })
                
                if batch_idx % 20 == 0:
                    # Save the model checkpoint to wandb
                    model_artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}",
                        type="model",
                        description="Trained text classification model",
                        metadata=dict(wandb.config)
                    )

                    # Save model to a file
                    t.save(self.model.probe.state_dict(), "model.pt")
                    model_artifact.add_file("model.pt")
                    wandb.log_artifact(model_artifact)


            # Calculate average loss for the epoch
            avg_train_loss = total_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with t.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    loss = loss_criterion(logits, labels)
                    val_loss += loss.item()
                    
                    _, predicted = t.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {accuracy:.2f}%")
            print("-" * 40)
            
            # Log epoch-level metrics
            wandb.log({
                "train/epoch_loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/accuracy": accuracy,
                "epoch": epoch + 1
            })

if __name__ == '__main__':
    assert device == "cuda"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Init wandb

    learning_rate = 1e-3
    batch_size = 32
    max_text_length = 6000
    positions_to_predict = 100

    wandb.init(
            project="qwen probe",
            config={
                "Learning Rate": learning_rate,
                "batch size": batch_size,
                "max text length": max_text_length,
                "positions to predict": positions_to_predict,
                }
    )

    # Set up models

    trainer = Trainer(model_name, learning_rate=learning_rate)
    tokenizer = trainer.model.tokenizer

    # Load in some data
    df = rs.read_ipc(Path("responses.arrow"))

    # Filter out anything where the answer doesn't end with the exact 
    # character (we want this for some metrics later)
    df = df.filter(
        rs.col("text")
        .str.slice(-1)
        .is_in(list("ABCDEFGHIJ"))
    )

    # Filter out realllly long responses.
    df = df.filter(rs.col("text").str.len_chars() <= max_text_length)
    
    # Training

    # The dataframe is already randomized, so we'll just split by the first 80%
    train_fraction = 0.8
    train_df = df[:int(len(df)*train_fraction)]
    val_df = df[int(len(df)*train_fraction):]

    # Create datasets
    train_dataset = TextDataset(train_df, tokenizer)
    val_dataset = TextDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)



    trainer.train(epochs=5, positions_to_predict=50)
