# This is decoder-only UT clock
import torch
import torch.nn as nn
from transformers_.gqa import GroupedQueryAttention
from .transition_mlp import FeedForwardTransition
from transformers_.rms_norm import RMSNorm
from typing import Optional, Tuple, Dict
class UniversalTransformersDecoderBlock(nn.Module):
    """One shared decoder block used recurrently across depth.
        Pre-norm reisdual structure:
        RMSNorm -> Masked Self-Attention -> Residual -> RMSNorm -> Transition -> Residual
        """
    def __init__(self, d_model: int, num_query_heads: int,num_kv_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dim = d_model
        self.num_heads = num_query_heads
        self.num_kv_heads = num_kv_heads

        #components
        self.norm1 = RMSNorm(d_model)
        self.masked_attn = GroupedQueryAttention(d_model,num_query_heads,num_kv_heads,dropout)
        self.norm2 = RMSNorm(d_model)
        self.transition = FeedForwardTransition(d_model,dropout=dropout,d_ff=d_ff)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None,
                cache: Optional[Tuple[torch.Tensor,torch.Tensor]]= None

                ):
        """
         x: [batch, seq_len, dim]
        causal_mask: [seq_len, seq_len] lower triangular mask
        cache: (past_k, past_v) for autoregressive decoding

        Returns:
            output: [batch, seq_len, dim]
            new_cache: (k, v) for next step
        """
        # 1. Masked Self-Attention sub-Block
        residual = x
        x_norm = self.norm1(x)

        #Apply masked attention with causal mask
        attn_out, new_cache = self.masked_attn(
            query = x_norm,
            key = x_norm,
            value = x_norm,
            attn_mask = causal_mask,
            is_causal = True, # Enable causal masking
            cache = cache
        )

        x = residual + attn_out

        # Transition sub-block
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.transition(x_norm)

        return x, new_cache
