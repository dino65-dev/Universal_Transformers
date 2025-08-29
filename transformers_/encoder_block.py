import torch.nn as nn
from multihead_attention import MultiHeadAttention
from residual_connection import ResidualConnection
from feed_forward_block import FeedForwardBlock
class EncoderBlock(nn.Module):
    """This block takes MultiheadAttention and FeedForward ,
    as well as dropout rate for residual connection"""

    def __init__(self,self_attention:MultiHeadAttention,feed_forward_block : FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        #storing the attention and ffn block
        self.attention_block = self_attention
        self.feed_forward_block = feed_forward_block
       # 2 residual connection with dropout
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x, src_mask):
        # Applying the first residual connection
        # Three 'x's corresponding to query, key, value
        x , _ = self.residual_connection[0](x, lambda x: self.attention_block(x,x,x,src_mask))

        # Applying the sceond residual connection
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x # output of EncoderBlock
