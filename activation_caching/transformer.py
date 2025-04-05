import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchtune.modules import RotaryPositionalEmbeddings


class RotaryMultiheadAttention(nn.Module):
    """
    MultiheadAttention with Rotary Position Embedding from torchtune
    """
    def __init__(self, d_model, num_heads, dropout=0.0, max_seq_len=1024):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Standard PyTorch MultiheadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Rotary embeddings from torchtune
        self.rotary_emb = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=max_seq_len
        )
    
    def _apply_rotary_to_qk(self, q, k, seq_len):
        # Extract the projections from MultiheadAttention's implementation
        # Reshape: [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, head_dim]
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        
        # Reshape back
        q = q.view(batch_size, -1, self.d_model)
        k = k.view(batch_size, -1, self.d_model)
        
        return q, k
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, is_causal=True):
        # Apply rotary embeddings
        seq_len = query.size(1)
        q, k = self._apply_rotary_to_qk(query, key, seq_len)
        
        # Now use the standard MultiheadAttention with rotary-embedded q and k
        # Set is_causal=True to ensure causal attention
        attn_output, attn_weights = self.mha(
            query=q,
            key=k,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal
        )
        
        return attn_output, attn_weights

class TransformerLayer(nn.Module):
    def __init__(self, d_res, num_heads, dropout=0.1, max_seq_len=1024):
        super().__init__()
        
        # Using standard PyTorch LayerNorm
        self.norm = nn.LayerNorm(d_res)
        
        # Self-attention with Rotary embeddings from torchtune
        self.self_attn = RotaryMultiheadAttention(
            d_model=d_res,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None, is_causal=True):
        # Pre-norm architecture
        x_norm = self.norm(x)
        
        # Apply attention with rotary embeddings and ensure causal attention
        attn_output, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal
        )
        
        # Residual connection
        return x + self.dropout(attn_output)

class SimpleTransformer(nn.Module):
    def __init__(self, d_input, d_res, num_layers, num_heads, num_classes=10, max_seq_len=1024, dropout=0.1, learning_rate=1e-3, device="cuda"):
        super().__init__()
        
        # Input embedding - linear projection from d_input to d_res
        self.embedding = nn.Linear(d_input, d_res, bias=False)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_res=d_res,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm_out = nn.LayerNorm(d_res)
        self.dropout = nn.Dropout(dropout)

        self.learning_rate = learning_rate
        self.input_layers = [59]

        self.mlp = nn.Sequential(
            nn.Linear(d_res, 4 * d_res),
            nn.GELU(),
            nn.Linear(4 * d_res, num_classes),
        )

        self.to(device, dtype=torch.bfloat16)
        
    def generate_causal_mask(self, seq_len, device):
        # Create causal attention mask for autoregressive models
        # In PyTorch's MultiheadAttention, the attn_mask should have True/False values
        # where False allows attention and True blocks it
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask
        
    def forward(self, x, mask=None, key_padding_mask=None, is_causal=True):
        batch_size, seq_len, _ = x.shape
        
        # Linear embedding from d_input to d_res
        x = self.embedding(x)
        
        # Apply dropout to embeddings
        x = self.dropout(x)

        if mask is None:
            mask = self.generate_causal_mask(seq_len, x.device)
        
        # For PyTorch's MultiheadAttention with is_causal=True,
        # we don't need to explicitly provide a mask, as it will create an optimal
        # causal mask internally. Only provide a mask if specified.
        
        # Apply transformer layers with rotary embeddings and causal attention
        for layer in self.layers:
            x = layer(x, attn_mask=mask, key_padding_mask=key_padding_mask, is_causal=is_causal)
        
        # Final normalization
        x = self.norm_out(x)

        # Run through MLP
        x = self.mlp(x)

        # Unembed to num classes
        return x

