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
from torch.optim.lr_scheduler import _LRScheduler
import math
from pathlib import Path
import threading
import queue
import torch as t
import numpy as np
from torchtune.modules import RotaryPositionalEmbeddings
from typing import List, Dict


CONFIG.set_default_api_key(Path("../nn_key.txt").read_text().strip())


device = "cuda" if t.cuda.is_available() else "cpu"


class WarmupCosineScheduler(_LRScheduler):
    """
    Implements a learning rate scheduler with linear warmup followed by cosine annealing.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for the linear warmup phase.
        max_steps (int): Total number of training steps.
        warmup_start_lr (float, optional): Initial learning rate for warmup. Default: 0.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_step (int, optional): The index of the last step. Default: -1.
    """
    
    def __init__(self, optimizer, warmup_steps, max_steps, warmup_start_lr=0.0, 
                 eta_min=0.0, last_step=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        # PyTorch uses 'last_epoch' internally, but we're using it as steps
        super(WarmupCosineScheduler, self).__init__(optimizer, last_step)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / 
                                (self.max_steps - self.warmup_steps))) / 2
                   for base_lr in self.base_lrs]


class LinearProbe(t.nn.Module):
    def __init__(self, in_features=8192, out_features=10, learning_rate=2e-4):
        super().__init__()
        # Simple linear probe for now
        in_features = in_features
        self.probe = t.nn.Linear(in_features, out_features, bias=True)
        self.learning_rate = learning_rate
        self.input_layers = [59]

        self.to(device, dtype=t.bfloat16)

    def forward(self, activations):
        return self.probe(activations)

class MlpProbe(t.nn.Module):
    def __init__(self, in_features=8192, hidden_dim=32, n_layers=1, out_features=10, learning_rate=2e-4):
        super().__init__()
        in_features = in_features
        self.layers = t.nn.Sequential(
                t.nn.Linear(in_features, hidden_dim, bias=True),
                t.nn.GELU(),
                *[
                    layer for _ in range(hidden_layers) for layer in [
                        t.nn.Linear(hidden_dim, hidden_dim),
                        t.nn.GELU(),
                        ]
                    ],
                t.nn.Linear(hidden_dim, out_features),
                )
                
        self.learning_rate = learning_rate
        self.input_layers = [59]

        self.to(device, dtype=t.bfloat16)

    def forward(self, activations):
        return self.layers(activations)
 
        

class TransformerEmbed(t.nn.Module):
    def __init__(self, 
                 model, 
                 input_layers: List[int] = [-1],
                 num_heads: int = 1,
                 out_features=10,
                 learning_rate=2e-5,
                 head_dim=32,
                 layer_norm=True,
                 ):
        super().__init__()

        self.model = model
        self.input_layers = input_layers
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.head_dim = head_dim
        self.pos_embed = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=5000)
        
        # We concatenate each probed layer into a single
        # input vector.
        in_features = model.config.hidden_size * len(input_layers)

        self.norm = None

        if layer_norm:
            self.norm = t.nn.RMSNorm(in_features)

        self.embed = t.nn.Linear(in_features, head_dim)

        self.W_attn = t.nn.Linear(head_dim, num_heads) # Get an affinity for each head

        # No unembed, straight to out-features
        self.W_OV = t.nn.Linear(head_dim, out_features * num_heads)
        
        self.to(dtype=self.model.dtype, device=device)

    def forward(self, layer_activations: t.Tensor):
        if self.norm is not None:
            acts = self.norm(layer_activations)
        else:
            acts = layer_activations.to(device)
            
        inputs = self.pos_embed(self.embed(acts).unsqueeze(-2)).squeeze(-2)

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


def calc_metrics(labels, logits):
    mask = labels != -100

    prob_last_token = 0
    prob_all_tokens = 0
    prob_last_10_tokens = 0

    for batch_logits, batch_labels, batch_mask in zip(logits, labels, mask):
        prob_last_token += batch_logits[batch_mask].softmax(dim=-1)[-1][batch_labels[batch_mask][-1].int()]
        prob_all_tokens += batch_logits[batch_mask].softmax(dim=-1)[batch_labels[batch_mask].int()].float().mean()
        prob_last_10_tokens += batch_logits[batch_mask].softmax(dim=-1)[-10:][batch_labels[batch_mask][-10:].int()].float().mean()

    prob_last_token /= logits.shape[0]
    prob_all_tokens /= logits.shape[0]
    prob_last_10_tokens /= logits.shape[0]

    return prob_last_token, prob_all_tokens, prob_last_10_tokens




class Trainer(t.nn.Module):
    def __init__(self, probe: TransformerProbe,
                 weight_decay=0.1,
                load_from=None,
                scheduler=lambda optim: None):
        super().__init__()

        self.probe = probe
        
        self.optimizer = t.optim.AdamW(probe.parameters(), lr=probe.learning_rate, weight_decay=weight_decay)
        self.scheduler = scheduler(self.optimizer)

        self.ignore_index = -100
        self.loss_criterion = t.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        wandb.watch(probe, self.loss_criterion, log="all", log_freq=10)

        self.layers_to_probe = probe.input_layers

        if load_from is not None:
            probe.load_state_dict(t.load(Path(load_from) / f"probe.pt", weights_only=True))


    def train(self, train_set: Dataset, test_set: Dataset = None, epochs=5, batch_size=10):
        step = 0
        sample = 0

        for epoch in tqdm.tqdm(range(epochs)):
            self.probe.train()
                
            train_loader = DataLoader(
                    train_set, batch_size, shuffle=True,
                    num_workers=3,
                    collate_fn=collate_fn,
                    pin_memory=True,
                )

            test_loader = DataLoader(
                    test_set, batch_size, shuffle=True,
                    num_workers=1,
                    collate_fn=collate_fn,
                )

            for i, batch in enumerate(train_loader):
                activations = batch["inputs"].to(device)
                labels = batch["labels"].long().to(device)
                
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

                if self.scheduler is not None:
                    self.scheduler.step()

                # Log batch-level metrics (every n batches)
                if sample % 200 <= batch_size:
                    prob_last_token, prob_all_tokens, prob_last_10_tokens = calc_metrics(labels, logits)

                    metrics = {
                        f"train/batch_loss": loss.float().item(),
                        f"train/batch": i,
                        f"train/samples": sample,
                        f"train/prob correct last token": prob_last_token.float(),
                        f"train/prob correct all tokens": prob_all_tokens.float(),
                        f"train/prob last 10 tokens": prob_last_10_tokens.float(),
                        f"train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    }

                    if test_set is not None and i % 500 <= batch_size:
                        test_losses = []
                        test_last_token, test_all_tokens, test_last_10 = [], [], []
                        for j, test_batch in enumerate(test_loader):
                            activations = test_batch["inputs"].to(device)
                            labels = test_batch["labels"].long().to(device)
                            logits = self.probe(activations)
                            test_loss = self.loss_criterion(logits.flatten(0, 1), labels.flatten(0, 1))

                            test_losses += [test_loss.item()]
                            
                            mets = calc_metrics(labels, logits)
                            test_last_token += [mets[0].item()]
                            test_all_tokens += [mets[1].item()]
                            test_last_10 += [mets[2].item()]

                        metrics["test/prob correct last token"] = np.mean(test_last_token)
                        metrics["test/prob correct all tokens"] = np.mean(test_all_tokens)
                        metrics["test/prob correct last 10 tokens"] = np.mean(test_last_10)
                        metrics["test/loss"] = np.mean(test_losses)


                    wandb.log(metrics, step=step)
                
                if i % 1000 == 0:
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
            num_heads=4,
            out_features=10,
            learning_rate=4e-3,
            full_tf_layers=0,
            residual_dim=32,
        )
#    probe = LinearProbe(learning_rate=2e-3)
   
    batch_size = 100
    weight_decay = 0.1

    warmup_steps = 500
    max_steps = 50_000

    positions_to_predict = 1000
    
    
    wandb.init(
            project="deepseek tf single probes 2",
            config={
                "batch size": batch_size,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "positions to predict": positions_to_predict,
                }
    )

    scheduler = lambda optim: WarmupCosineScheduler(optimizer=optim, warmup_steps=warmup_steps,
            max_steps=max_steps,
            warmup_start_lr=1e-8,
            eta_min=1e-8
            )

    trainer = Trainer(probe=probe, weight_decay=weight_decay, scheduler=scheduler)

    train_set = SavedActivationDataset("~/tf/cotprobe/activations", df, positions_to_predict=positions_to_predict)
    test_set = SavedActivationDataset("~/tf/cotprobe/test_act", df, positions_to_predict=100)
    
    trainer.train(batch_size=batch_size, train_set=train_set, test_set=test_set)

