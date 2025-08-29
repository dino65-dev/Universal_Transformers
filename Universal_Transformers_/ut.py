# Universal Transformers
import torch.nn as nn
import math
import torch
from ut_block import UniversalTransformerBlock
from transformers_.rms_norm import RMSNorm
class UniversalTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 512, max_seq_len: int = 1024, num_heads: int = 8, num_kv_heads: int = 4, T: int = 6, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.T = T  # Number of recurrent steps (depth)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.block = UniversalTransformerBlock(dim, num_heads, num_kv_heads, dropout)
        self.final_norm = RMSNorm(dim)

        # Learnable sinusoidal positional and timestep embeddings
        self.register_buffer("pos_emb", self._sinusoidal_embeddings(max_seq_len, dim))
        self.register_buffer("time_emb", self._sinusoidal_embeddings(T + 1, dim))  # +1 for indexing

    def _sinusoidal_embeddings(self, max_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        emb = torch.zeros(max_len, dim)
        emb[:, 0::2] = torch.sin(pos * div_term)
        emb[:, 1::2] = torch.cos(pos * div_term)
        return emb.unsqueeze(0)  # [1, max_len, dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len] (token indices)
        batch, seq_len = x.shape
        x = self.embedding(x)  # [B, S, D]

        # Add positional embeddings once
        x = x + self.pos_emb[:, :seq_len, :]

        # Recurrent loop over T steps
        for t in range(1, self.T + 1):  # t from 1 to T
            # Add timestep embedding
            time_emb_t = self.time_emb[:, t - 1, :].unsqueeze(1).expand(batch, seq_len, self.dim)  # [B, S, D]
            x = x + time_emb_t

            # Apply the shared block
            x = self.block(x)

        x = self.final_norm(x)
        return x  # Output: [B, S, D] (refined representations)
