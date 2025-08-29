# Universal Transformers Block
import torch.nn as nn
import torch
from transition_mlp import FeedForwardTransition
from transformers_.rms_norm import RMSNorm
from transformers_.gqa import GroupedQueryAttention

class UniversalTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_kv_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.transition = FeedForwardTransition(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-block with residual
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = res + x

        # Transition sub-block with residual
        res = x
        x = self.norm2(x)
        x = self.transition(x)
        x = res + x

        return x
