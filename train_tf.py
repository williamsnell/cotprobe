from itertools import product
from nnsight import CONFIG
from nnsight import LanguageModel
from typing import List, Dict
import polars as pl
import wandb
import tqdm
import torch as t
from pathlib import Path
import threading
import queue
import torch as t
from typing import List, Dict


CONFIG.set_default_api_key(Path("nn_key.txt").read_text().strip())


model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

model = LanguageModel(model_name)

df = pl.read_ipc("responses_70b.arrow")
train = df[:int(len(df)*0.8)]
test = df[int(len(df)*0.8):]


def get_activations(model, prompts, layers: List[int], tokens=1000):
    hidden_states = {}

    is_remote = True # if model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" else False

    with model.trace(prompts, remote=is_remote) as runner:
        for layer in layers:
            hidden_states[layer] = model.model.layers[layer].output[0][:, :-tokens].clone().save()
    
    return hidden_states



device = "cuda" if t.cuda.is_available() else "cpu"


class TransformerEmbed(t.nn.Module):
    def __init__(self, 
                 model, 
                 input_layers: List[int] = [-1],
                 num_heads: int = 1,
                 out_features=10,
                 learning_rate=2e-5,
                 positions_to_predict=None):
        super().__init__()

        self.model = model
        self.input_layers = input_layers
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.positions_to_predict = positions_to_predict

        # We concatenate each probed layer into a single
        # input vector.
        in_features = model.config.hidden_size * len(input_layers)

        self.W_attn = t.nn.Linear(in_features, num_heads) # Get an affinity for each head

        # No unembed, straight to out-features
        self.W_OV = t.nn.Linear(in_features, out_features * num_heads)
        
        self.to(dtype=self.model.dtype, device=device)

    def forward(self, layer_activations: Dict[int, t.Tensor]):
        inputs = t.concat([layer_activations[input_layer].clone().detach() for input_layer in self.input_layers], dim=-1 # [batch, seq_len, 1]
                          ).to(device)

        # Calculate affinity (we have the same query every time)
        kq = self.W_attn(inputs).unsqueeze(-2) # [batch, seq_len, 1, num_heads]

        seq_len, num_heads = kq.shape[-3:-1]

        # Create attention pattern
        unnormalized_attn = kq.expand(-1, -1, seq_len, -1).permute(0, 3, 1, 2) # [batch, num_heads, seq_len, seq_len]
        mask = t.triu(t.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        masked_unnorm_attn = unnormalized_attn + mask # [batch, num_heads, seq_len, seq_len]
        attn_pattern = masked_unnorm_attn.softmax(dim=-1) # [batch, num_heads, seq_len, seq_len]

        # Effectively a linear probe into each sequence position.
        ov = self.W_OV(inputs).unflatten(-1, (num_heads, -1)).swapaxes(-2, -3) # [batch, seq_len, num_heads, out_features]

        # Weight each sequence probe by normed affinity,
        # and sum across heads
        out = (attn_pattern @ ov).sum(dim=-3) # [batch, seq_len, out_features]
           
        return out

class TransformerProbe(t.nn.Module):
    def __init__(self, 
                 model, 
                 input_layers: List[int] = [-1],
                 num_heads: int = 1,
                 out_features=10,
                 learning_rate=2e-5,
                 positions_to_predict=None,
                 full_tf_layers: int = 0,
                 residual_dim=0,
                 ):

        super().__init__()

        self.input_layers = input_layers
        self.num_heads = num_heads
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.positions_to_predict = positions_to_predict

        # If we don't have any following transformer blocks, we just directly unembed
        self.residual_dim = out_features if full_tf_layers == 0 else residual_dim
        if full_tf_layers != 1 and residual_dim == 0:
            raise ValueError("Must specify a residual_dim if there are 1 or more transformer layers.")

        self.mlp_dim = 4 * residual_dim

        self.act = t.nn.GELU()

        self.layers = [
                TransformerEmbed(
                    model, input_layers, num_heads, self.residual_dim, learning_rate, positions_to_predict
                    )
                ]

        for layer in range(full_tf_layers):
            self.layers += [t.nn.TransformerEncoderLayer(self.residual_dim, self.num_heads, self.mlp_dim, self.act)]

        # Unembed, if necessary
        if self.residual_dim != out_features:
            self.layers += [t.nn.Linear(self.residual_dim, out_features)]


        self.sequence = t.nn.Sequential(*self.layers)
                

    def forward(self, layer_activations: Dict[int, t.Tensor]):
        return self.sequence(layer_activations)



    def from_text(self, text: str):
        activations = get_activations(self.model, text, self.input_layers)

        return self.forward(activations)        


class ActivationManager:
    """Manages the retrieval of activations with controlled concurrency for NNSight compatibility"""
    def __init__(self, model, layers_to_probe, num_workers=1, queue_size=4):
        self.model = model
        self.layers_to_probe = layers_to_probe
        self.activation_queue = queue.Queue(maxsize=queue_size)
        self.batch_queue = queue.Queue(maxsize=queue_size) 
        self.num_workers = num_workers
        self.stop_event = threading.Event()
        self.worker_threads = []
        self.trace_lock = threading.Lock()  # Lock for trace operations
        self.error = None

    def start_workers(self, data_iterator, batch_size):
        """Start workers with a controlled approach to avoid NNSight concurrency issues"""
        self.stop_event.clear()
        self.error = None
        
        # Start the batch distributor thread
        self.batch_distributor = threading.Thread(
            target=self._batch_distributor, 
            args=(data_iterator, batch_size),
            daemon=True
        )
        self.batch_distributor.start()
        
        # Start worker threads
        self.worker_threads = []
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._activation_worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
    
    def _batch_distributor(self, data_iterator, batch_size):
        """Distributes batches to worker threads"""
        try:
            num_batches = len(data_iterator) // batch_size
            for batch_idx in range(num_batches):
                if self.stop_event.is_set():
                    break
                
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_info = {
                    "batch_idx": batch_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "data": data_iterator  # Pass reference to data
                }
                
                # This will block if all workers are busy and queue is full
                self.batch_queue.put(batch_info)
                
            # Signal end of batches with None
            for _ in range(self.num_workers):
                self.batch_queue.put(None)
                
        except Exception as e:
            self.error = e
            print(f"Error in batch distributor: {e}")
            self.stop_event.set()
        
    def _activation_worker(self, worker_id):
        """Worker function that safely retrieves activations"""
        try:
            while not self.stop_event.is_set():
                # Get next batch assignment
                try:
                    batch_info = self.batch_queue.get(timeout=5)
                except queue.Empty:
                    continue
                    
                if batch_info is None:  # Signal to stop
                    self.batch_queue.task_done()
                    break
                
                batch_idx = batch_info["batch_idx"]
                start_idx = batch_info["start_idx"]
                end_idx = batch_info["end_idx"]
                data = batch_info["data"]
                
                try:
                    # Extract the data for this batch
                    prompts = data["text"][start_idx:end_idx].to_list()
                    
                    # Make local copy of tokenized data to avoid thread issues
                    tokenized = self.model.tokenizer(prompts, padding=True, return_tensors='pt')
                    
                    # Get activations from the model - with lock for thread safety
                    with self.trace_lock:
                        try:
                            hidden_states = self._get_activations(prompts)
                        except Exception as e:
                            print(f"Worker {worker_id} - Error getting activations: {e}")
                            self.batch_queue.task_done()
                            continue
                    
                    # Put the result in the queue
                    batch_data = {
                        "prompts": prompts,
                        "tokenized": tokenized,
                        "activations": hidden_states,
                        "labels": data["numeric label"][start_idx:end_idx],
                        "batch_idx": batch_idx
                    }
                    
                    self.activation_queue.put(batch_data)
                    
                except Exception as e:
                    print(f"Worker {worker_id} - Error processing batch: {e}")
                
                self.batch_queue.task_done()
                
        except Exception as e:
            self.error = e
            print(f"Worker {worker_id} - Fatal error: {e}")
            self.stop_event.set()
            
    def _get_activations(self, prompts):
        """Get activations for the specified layers with robust error handling"""
        hidden_states = {}
        is_remote = True
        
        try:
            with self.model.trace(prompts, remote=is_remote) as runner:
                for layer in self.layers_to_probe:
                    try:
                        hidden_states[layer] = self.model.model.layers[layer].output[0].save()
                    except Exception as e:
                        print(f"Error saving layer {layer} output: {e}")
                        # Create an empty tensor as a fallback
                        # This allows training to continue even if one layer fails
                        hidden_states[layer] = t.zeros(1)
        except Exception as e:
            print(f"Error in model.trace: {e}")
            raise
        
        return hidden_states
    
    def get_next_batch(self, timeout=60):
        """Get the next batch of activations with timeout"""
        if self.error:
            raise self.error
            
        try:
            batch_data = self.activation_queue.get(timeout=timeout)
            self.activation_queue.task_done()
            return batch_data
        except queue.Empty:
            raise TimeoutError("Timeout waiting for activations")
    
    def stop(self):
        """Stop all workers and clean up resources"""
        self.stop_event.set()
        
        # Drain the queues to unblock any waiting threads
        try:
            while True:
                self.batch_queue.get_nowait()
                self.batch_queue.task_done()
        except queue.Empty:
            pass
            
        try:
            while True:
                self.activation_queue.get_nowait()
                self.activation_queue.task_done()
        except queue.Empty:
            pass
        
        # Wait for threads to finish with a timeout
        for thread in self.worker_threads:
            thread.join(timeout=2)
        
        if self.batch_distributor.is_alive():
            self.batch_distributor.join(timeout=2)



class Trainer(t.nn.Module):
    def __init__(self, probes: List[TransformerProbe], language_model=model, num_workers=1,
                load_from=None):
        super().__init__()
        self.language_model = language_model
        self.probes = probes
        
        self.optimizers = [t.optim.AdamW(probe.parameters(), lr=probe.learning_rate)
                          for probe in probes]

        self.ignore_index = -100
        self.loss_criterion = t.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.layers_to_probe = []
        for probe in self.probes:
            wandb.watch(probe, self.loss_criterion, log="all", log_freq=10)
            self.layers_to_probe += probe.input_layers

        self.layers_to_probe = list(set(self.layers_to_probe))
        
        # Create the activation manager with multiple workers
        self.activation_manager = ActivationManager(
            self.language_model, 
            self.layers_to_probe,
            num_workers=num_workers,
            queue_size=num_workers * 2  # Larger queue to buffer more activations
        )

        if load_from is not None:
            for i, probe in enumerate(self.probes):
                probe.load_state_dict(t.load(Path(load_from) / f"probe-{i}.pt", weights_only=True))


    def train(self, epochs=5, batch_size=10, train_set=train):
        step = 0
        
        for epoch in tqdm.tqdm(range(epochs)):
            for probe in self.probes:
                probe.train()
                
            total_loss = [0] * len(self.probes)

            # Shuffle the training set
            train_set = train_set.sample(fraction=1.0, shuffle=True)
            
            # Start the activation worker for this epoch
            self.activation_manager.start_workers(train_set, batch_size)
            
            num_batches = len(train_set) // batch_size
            for _ in range(num_batches):
                # Get the next batch data including activations
                try:
                    batch_data = self.activation_manager.get_next_batch(timeout=2400)  # 5 minutes timeout
                except queue.Empty:
                    tqdm.tqdm.write("Timeout waiting for activations")
                    continue
                    
                prompts = batch_data["prompts"]
                tokenized = batch_data["tokenized"]
                activations = batch_data["activations"]
                batch_labels = batch_data["labels"]
                
                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']
                
                labels = t.tensor(batch_labels).to(device).unsqueeze(-1).repeat(1, input_ids.shape[-1])

                step += 1

                for i, (probe, optim) in enumerate(zip(self.probes, self.optimizers)):
                    optim.zero_grad()
                    # Make predictions
                    logits = probe(activations)

                    # Mask out any pad tokens
                    logits[attention_mask == 0] = self.ignore_index

                    # Allow only predicting the last n tokens
                    if probe.positions_to_predict is not None:
                        logits = logits[:, -probe.positions_to_predict:]
                        labels = labels[:, -probe.positions_to_predict:]
    
                    loss = self.loss_criterion(logits.flatten(0, 1), labels.flatten(0, 1))
    
                    # tqdm.tqdm.write(f"Loss-{i}: {loss.item()}")
    
                    loss.backward()
                    optim.step()
    
                    total_loss[i] += loss.item()
    
                    # Log batch-level metrics (every n batches)
                    batch = batch_data["batch_idx"]
                    if batch % 1 == 0:
                        wandb.log({
                            f"train/batch_loss/{i}": loss.item(),
                            f"train/batch/{i}": batch,
                            f"train/global_step/{i}": epoch * len(train_set) + batch,
                            f"train/prob correct last token/{i}": logits[:, -1].softmax(dim=-1)[range(logits.shape[0]), labels[:, -1].int()].mean(),
                            f"train/prob correct all tokens/{i}": logits.softmax(dim=-1).flatten(0, 1)[range(logits.shape[0] * logits.shape[1]), labels.flatten(0, 1)].mean(),
                        }, step=step)
                    
                    if batch % 20 == 0:
                        # Save the model checkpoint to wandb
                        model_artifact = wandb.Artifact(
                            name=f"model-{wandb.run.id}-{i}",
                            type="model",
                            description="Trained hidden-layer probe.",
                            metadata=dict(wandb.config)
                        )
    
                        # Save model to a file
                        t.save(probe.state_dict(), f"probe-{i}.pt")
                        model_artifact.add_file(f"probe-{i}.pt")
                        wandb.log_artifact(model_artifact)
            
            # Stop the activation worker at the end of the epoch
            self.activation_manager.stop()

            # Save the final states
            for i, probe in enumerate(self.probes):
                t.save(probe.state_dict(), f"probe-{i}.pt")

if __name__ == "__main__":
    model = LanguageModel(model_name)
    
    probes = [    
        TransformerProbe(
            model, 
            input_layers=[59],
            num_heads=num_heads,
            out_features=10,
            learning_rate=learning_rate,
            positions_to_predict=None,
            full_tf_layers=full_tf_layers,
            residual_dim=32,
            include_MLP=True
        )

        for (learning_rate, num_heads, full_tf_layers) in product(
            [2e-4, 2e-5, 2e-6], [1, 4, 16], [0, 1, 2],
        )
    ]
    
    max_text_length = "4096 Tokens"
    batch_size = 11
    
    wandb.init(
            project="deepseek probes",
            name="5 - broad sweep cont.",
            config={
                "batch size": batch_size,
                "max text length": max_text_length,
                "probes": str(probes),
                }
    )
    
    trainer = Trainer(probes=probes)
    
    trainer.train(batch_size=batch_size)
    
    
    
