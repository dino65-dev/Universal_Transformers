from transformer import Transformer
from input_embeddings import InputEmbeddings
from postional_encoding import PostionalEncoding
from encoder_block import EncoderBlock
from encoder import Encoder
from decoder_block import DecoderBlock
from decoder import Decoder
from gqa import GroupedQueryAttention
from feed_forward_block import FeedForwardBlock
from projection_layer import ProjectionLayer
import torch.nn as nn
# Building and initializing Transformer (decoder_only)
# N --> stack of decoder , h -> head, d_ff --> neurons of feed forward , kv_h --> key&value heads
def build_transformer(src_vocab_size : int, tgt_vocab_size : int,src_seq_len: int, tgt_seq_len: int, d_model : int = 512, N: int = 6, h: int = 8, kv_h : int = 4, dropout: float = 0.1, d_ff: int = 2048 ):

    # Creating Embdding layers
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

    # Creating Postitional Encoding layers
    tgt_pos = PostionalEncoding(d_model,tgt_seq_len,dropout)

    # Creating DecoderBlocks
    decoder_blocks = [] # Initial list of empty decoderBlocks

    for _ in range(N): # Iterating 'N' times to create 'N' DecoderBlocks (N = 6)
        decoder_self_attention_block = GroupedQueryAttention(d_model,h,kv_h,dropout) # grouped_query_attention
        decoder_cross_attention_block = GroupedQueryAttention(d_model,h,kv_h,dropout) # cross attention
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)

        # combining layers into a DecoderBlock
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)  # Appending DecoderBlock to the list of DecoderBlocks

    # Creating the Decoder by using DecoderBlocks Lists
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Creating projection Layer
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)  # Map the output of Decoder to the Target Vocabulary Space

    # Creating Transformer by Combining everything above
    transformer = Transformer(None,decoder,None,tgt_embed,None,tgt_pos,projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer # Assembled and initialized Transformer. Ready to be trained and validated!
