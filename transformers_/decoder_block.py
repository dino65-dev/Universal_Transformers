import torch.nn as nn
from feed_forward_block import FeedForwardBlock
from residual_connection import ResidualConnection
from gqa import GroupedQueryAttention
class DecoderBlock(nn.Module):
    """Decoder Block takes in Two MultiHeadAttention :
         one is self-attention another is cross attention
        It also takes one feed-forward block and dropout rate """

    def __init__(self, masked_attention_block : GroupedQueryAttention, cross_attention_block : GroupedQueryAttention, feed_forward_block: FeedForwardBlock, dropout: float  ) -> None:
        super().__init__()
        self.masked_attention = masked_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        # 3 residual connetion with dropout rate
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output_key, encoder_output_value, src_mask, tgt_mask, cache=None):
        # Self-attention block
        x, self_attn_cache = self.residual_connection[0](
            x,
            lambda x: self.masked_attention(
                query=x,
                key=x,
                value=x,
                attn_mask=tgt_mask,
                is_causal=True,
                cache=None if cache is None else cache[0]  # First element for self-attention
            )
        )

        # Cross-attention block
        x, cross_attn_cache = self.residual_connection[1](
            x,
            lambda x: self.cross_attention(
                query=x,
                key=encoder_output_key,
                value=encoder_output_value,
                attn_mask=src_mask,
                cache=None if cache is None else cache[1]  # Second element for cross-attention
            )
        )

        # Feed forward block
        x = self.residual_connection[2](x, self.feed_forward)
        return x, (self_attn_cache, cross_attn_cache)  # Return tuple of caches
