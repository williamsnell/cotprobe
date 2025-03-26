from nnsight import CONFIG
from dataclasses import dataclass
from nnsight import LanguageModel
from typing import List, Dict
import polars as pl
import wandb
import re
import tqdm
import torch as t
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import threading
import queue
import time
import torch as t
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor


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


class Probe(t.nn.Module):
    def __init__(self, 
                 model, 
                 input_layers: List[int] = [-1],
                 hidden_layers: int = 0, hidden_dim: int = 0, 
                 out_features=10,
                 learning_rate=2e-5,
                 positions_to_predict=1000):
        super().__init__()

        self.model = model
        self.input_layers = input_layers
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.positions_to_predict = positions_to_predict

        # We concatenate each probed layer into a single
        # input vector.
        in_features = model.config.hidden_size * len(input_layers)

        # Linear Probe
        if hidden_layers == 0:
            self.probe = t.nn.Linear(in_features, out_features)
        # MLP
        else:
            self.probe = t.nn.Sequential(
                t.nn.Linear(in_features, hidden_dim),
                t.nn.ReLU(),
                *[
                    layer for i in range(hidden_layers) for layer in [
                        t.nn.Linear(hidden_dim, hidden_dim),
                        t.nn.ReLU(),
                    ]
                ],
                t.nn.Linear(hidden_dim, out_features)
            )                  

        self.to(dtype=self.model.dtype, device=device)

    def forward(self, layer_activations: Dict[int, t.tensor]):
        inputs = t.concat([layer_activations[input_layer].clone().detach() for input_layer in self.input_layers]).to(device)
        return self.probe(inputs)

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
                        hidden_states[layer] = torch.zeros(1)
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
    def __init__(self, probes: List[Probe], language_model=model, num_workers=1,
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


    def train(self, epochs=5, batch_size=10, positions_to_predict=None, train_set=train):
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
        # Linear
        
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-9),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-8),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-7),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-6),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-5),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-4),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-3),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-2),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-1),
    
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-9),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-8),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-7),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-6),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-5),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-4),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-3),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-2),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-1),
    
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-9),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-8),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-7),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-6),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-5),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-4),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-3),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-2),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-1),
    
        # MLP
    
        # 80-wide hidden
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-9),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-8),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-7),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-6),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-5),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, learning_rate=2e-4),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, learning_rate=2e-3),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, learning_rate=2e-2),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, learning_rate=2e-1),
    
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-9),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-8),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-7),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-6),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-5),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-4),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-3),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-2),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-1),
    
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-9),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-8),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-7),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-6),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-5),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-4),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-3),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-2),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-1),
    
        # 160-wide hidden
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-9),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-8),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-7),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-6),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-5),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, learning_rate=2e-4),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, learning_rate=2e-3),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, learning_rate=2e-2),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, learning_rate=2e-1),
    
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-9),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-8),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-7),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-6),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-5),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-4),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-3),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-2),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-1),
    
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-9),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-8),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-7),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-6),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-5),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-4),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-3),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-2),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-1),
    
    
        # Pos to predict = 50
        # Linear
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-9, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-8, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-7, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-6, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-5, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-4, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-3, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-2, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=0, out_features=10, learning_rate=2e-1, positions_to_predict=50),
    
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-9, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-8, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-7, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-6, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-5, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-4, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-3, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-2, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=0, out_features=10, learning_rate=2e-1, positions_to_predict=50),
    
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-9, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-8, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-7, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-6, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-5, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-4, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-3, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-2, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=0, out_features=10, learning_rate=2e-1, positions_to_predict=50),
    
        # MLP
    
        # 80-wide hidden
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-9, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-8, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-7, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-6, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, out_features=10, learning_rate=2e-5, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, learning_rate=2e-4, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, learning_rate=2e-3, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, learning_rate=2e-2, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=80, learning_rate=2e-1, positions_to_predict=50),
    
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-9, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-8, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-7, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-6, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-5, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-4, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-3, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-2, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=80, learning_rate=2e-1, positions_to_predict=50),
    
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-9, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-8, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-7, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-6, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-5, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-4, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-3, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-2, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=80, learning_rate=2e-1, positions_to_predict=50),
    
        # 160-wide hidden
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-9, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-8, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-7, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-6, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, out_features=10, learning_rate=2e-5, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, learning_rate=2e-4, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, learning_rate=2e-3, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, learning_rate=2e-2, positions_to_predict=50),
        Probe(model, input_layers=[79], hidden_layers=1, hidden_dim=160, learning_rate=2e-1, positions_to_predict=50),
    
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-9, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-8, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-7, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-6, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-5, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-4, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-3, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-2, positions_to_predict=50),
        Probe(model, input_layers=[59], hidden_layers=1, hidden_dim=160, learning_rate=2e-1, positions_to_predict=50),
    
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-9, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-8, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-7, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-6, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-5, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-4, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-3, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-2, positions_to_predict=50),
        Probe(model, input_layers=[39], hidden_layers=1, hidden_dim=160, learning_rate=2e-1, positions_to_predict=50),
        
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
    
    trainer = Trainer(probes=probes, load_from="trained")
    
    trainer.train(batch_size=batch_size)
    
    
    
