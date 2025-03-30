from load_activations import collate_fn, SavedActivationDataset
from itertools import product
from nnsight import CONFIG
from nnsight import LanguageModel
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
import polars as pl
import wandb
import tqdm
import torch as t
from pathlib import Path
import threading
import queue
import torch as t
from typing import List, Dict


CONFIG.set_default_api_key(Path("../nn_key.txt").read_text().strip())


device = "cuda" if t.cuda.is_available() else "cpu"


class TransformerEmbed(t.nn.Module):
    def __init__(self, 
                 model, 
                 input_layers: List[int] = [-1],
                 num_heads: int = 1,
                 out_features=10,
                 learning_rate=2e-5,
                 ):
        super().__init__()

        self.model = model
        self.input_layers = input_layers
        self.out_features = out_features
        self.learning_rate = learning_rate

        # We concatenate each probed layer into a single
        # input vector.
        in_features = model.config.hidden_size * len(input_layers)

        self.W_attn = t.nn.Linear(in_features, num_heads) # Get an affinity for each head

        # No unembed, straight to out-features
        self.W_OV = t.nn.Linear(in_features, out_features * num_heads)
        
        self.to(dtype=self.model.dtype, device=device)

    def forward(self, layer_activations: t.Tensor):
        inputs = layer_activations.to(device)

        # Calculate affinity (we have the same query every time)
        kq = self.W_attn(inputs).unsqueeze(-2) # [batch, seq_len, 1, num_heads]

        seq_len, num_heads = kq.shape[-3], kq.shape[-1]

        # Create attention pattern
        unnormalized_attn = kq.expand(-1, -1, seq_len, -1).permute(0, 3, 1, 2) # [batch, num_heads, seq_len, seq_len]
        mask = t.triu(t.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1).to(device=device, dtype=self.model.dtype)
        masked_unnorm_attn = unnormalized_attn + mask # [batch, num_heads, seq_len, seq_len]
        attn_pattern = masked_unnorm_attn.softmax(dim=-1) # [batch, num_heads, seq_len, seq_len]

        # Effectively a linear probe into each sequence position.
        ov = self.W_OV(inputs).unflatten(-1, (num_heads, -1)).swapaxes(-2, -3) # [batch, seq_len, num_heads, out_features]

        # Weight each sequence probe by normed affinity,
        # and sum across heads
        out = (attn_pattern @ ov).sum(dim=-3) # [batch, seq_len, out_features]
           
        return out


class CausalTransformerEncoderLayer(t.nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", batch_first=True):
        super().__init__(d_model=d_model, nhead=nhead, 
                         dim_feedforward=dim_feedforward, 
                         dropout=dropout, activation=activation,
                         batch_first=batch_first)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if src_mask is None:
            src_mask = t.nn.Transformer.generate_square_subsequent_mask(src.shape[1], device=src.device, dtype=src.dtype)
        # Always use causal attention
        return super().forward(src, src_mask, src_key_padding_mask, is_causal=True)


class TransformerProbe(t.nn.Module):
    def __init__(self, 
                 model, 
                 input_layers: List[int] = [-1],
                 num_heads: int = 1,
                 out_features=10,
                 learning_rate=2e-5,
                 full_tf_layers: int = 0,
                 residual_dim=0,
                 ):

        super().__init__()

        self.model = model
        self.input_layers = input_layers
        self.num_heads = num_heads
        self.out_features = out_features
        self.learning_rate = learning_rate

        # If we don't have any following transformer blocks, we just directly unembed
        self.residual_dim = out_features if full_tf_layers == 0 else residual_dim
        if full_tf_layers > 0 and residual_dim == 0:
            raise ValueError("Must specify a residual_dim if there are 1 or more transformer layers.")

        self.mlp_dim = 4 * residual_dim

        self.act = t.nn.GELU()

        self.layers = [
                TransformerEmbed(
                    model, input_layers, num_heads, self.residual_dim, learning_rate
                    )
                ]

        for layer in range(full_tf_layers):
            self.layers += [CausalTransformerEncoderLayer(self.residual_dim, self.num_heads, self.mlp_dim, activation=self.act, batch_first=True)]

        # Unembed, if necessary
        if self.residual_dim != out_features:
            self.layers += [t.nn.Linear(self.residual_dim, out_features)]


        self.sequence = t.nn.Sequential(*self.layers)

        self.to(device=device, dtype=self.model.dtype)
                

    def forward(self, layer_activations: Dict[int, t.Tensor]):
        return self.sequence(layer_activations)

    def from_text(self, text: str):
        activations = get_activations(self.model, text, self.input_layers)

        return self.forward(activations)        


class Trainer(t.nn.Module):
    def __init__(self, probe: TransformerProbe,
                 weight_decay=0.1,
                load_from=None):
        super().__init__()

        self.probe = probe
        
        self.optimizer = t.optim.AdamW(probe.parameters(), lr=probe.learning_rate, weight_decay=weight_decay)

        self.ignore_index = -100
        self.loss_criterion = t.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        wandb.watch(probe, self.loss_criterion, log="all", log_freq=10)

        self.layers_to_probe = probe.input_layers

        if load_from is not None:
            probe.load_state_dict(t.load(Path(load_from) / f"probe.pt", weights_only=True))


    def train(self, train_set: Dataset, epochs=5, batch_size=10):
        step = 0
        sample = 0

        for epoch in tqdm.tqdm(range(epochs)):
            self.probe.train()
                
            train_loader = DataLoader(
                    train_set, batch_size, shuffle=True,
                    num_workers=3,
                    collate_fn=collate_fn,
                )

            for i, batch in enumerate(train_loader):
                activations = batch["inputs"]
                labels = batch["labels"].long()
                
                step += 1
                sample += activations.shape[0]

                self.optimizer.zero_grad()
                # Make predictions
                logits = self.probe(activations)

                # Padding tokens should already be pre-masked
                loss = self.loss_criterion(logits.flatten(0, 1), labels.flatten(0, 1))

                # tqdm.tqdm.write(f"Loss-{i}: {loss.item()}")

                loss.backward()
                self.optimizer.step()

                # Log batch-level metrics (every n batches)
                if i % 1 == 0:
                    mask = labels != -100

                    prob_last_token = 0
                    prob_all_tokens = 0

                    for batch_logits, batch_labels, batch_mask in zip(logits, labels, mask):
                        prob_last_token += batch_logits[batch_mask].softmax(dim=-1)[-1][batch_labels[batch_mask][-1].int()]
                        prob_all_tokens += batch_logits[batch_mask].softmax(dim=-1)[batch_labels[batch_mask].int()].mean()


                    prob_last_token /= logits.shape[0]
                    prob_all_tokens /= logits.shape[0]


                    wandb.log({
                        f"train/batch_loss": loss.float().item(),
                        f"train/batch": i,
                        f"train/global_step": step,
                        f"train/samples": sample,
                        f"train/prob correct last token": prob_last_token.float(),
                        f"train/prob correct all tokens": prob_all_tokens.float(),
                        f"train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    }, step=step)
                
                if i % 100 == 0:
                    # Save the model checkpoint to wandb
                    model_artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}",
                        type="model",
                        description="Trained hidden-layer probe.",
                        metadata=dict(wandb.config)
                    )

                    # Save model to a file
                    t.save(self.probe.state_dict(), f"probe.pt")
                    model_artifact.add_file(f"probe.pt")
                    wandb.log_artifact(model_artifact)
        
            # Save the final states
            t.save(probe.state_dict(), f"probe.pt")

if __name__ == "__main__":
    model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")

    df = pl.read_ipc("responses_70b_with_sizes.arrow")

    probe = TransformerProbe(
            model, 
            input_layers=[59],
            num_heads=2,
            out_features=10,
            learning_rate=1e-3,
            full_tf_layers=0,
        )
   
    batch_size = 20
    weight_decay = 0.1
    
    wandb.init(
            project="deepseek tf single probes",
            name="1.",
            config={
                "batch size": batch_size,
                "weight_decay": weight_decay,
                }
    )
   
    trainer = Trainer(probe=probe, weight_decay=weight_decay)

    train_set = SavedActivationDataset("activations", df)
    
    trainer.train(batch_size=batch_size, train_set=train_set)
    

