import torch.nn as nn
from typing import Optional  # Add this import
from encoder import Encoder
from decoder import Decoder
from input_embeddings import InputEmbeddings
from projection_layer import ProjectionLayer
from postional_encoding import PostionalEncoding

class Transformer(nn.Module):

    """This takes in the encoder and decoder, as well the embeddings for the source
     and target language. It also takes in the postional encoding for the source and target language,
      as well as projection layer """

    def __init__(self,
                 encoder: Optional[Encoder] = None,
                 decoder: Optional[Decoder] = None,
                 src_embed: Optional[InputEmbeddings] = None,
                 tgt_embed: Optional[InputEmbeddings] = None,
                 src_pos: Optional[PostionalEncoding] = None,
                 tgt_pos: Optional[PostionalEncoding] = None,
                 projection_layer: Optional[ProjectionLayer] = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # Encoder
    def encode(self,src,src_mask):
        if self.src_embed is None or self.src_pos is None or self.encoder is None:
            raise ValueError("Encoder components are not initialized. This is a decoder-only model.")
        src = self.src_embed(src) # Applying source embeddings to the input source language
        src = self.src_pos(src) # Applying source positional encoding to the source embeddings
        return self.encoder(src,src_mask) # Returning the source embeddings plus a source mask to prevent attention to certain elements

    #Decoder
    # tgt_embed --> right shifted embedding
    #src_embed --> tensor of embed
    #src_pos --> normal pos of embbedings(encoder)
    #tgt_pos --> same formula but for decoder input
    def decode(self, encoder_output_key, encoder_output_value, src_mask, tgt , tgt_mask, layer_caches = None):
        if self.tgt_embed is None or self.tgt_pos is None or self.decoder is None:
            raise ValueError("Decoder components are not properly initialized.")
        tgt = self.tgt_embed(tgt) # Applying target embeddings to the input target language (tgt)
        tgt = self.tgt_pos(tgt) # Applying target positional encoding to the target embeddings

        output, new_caches = self.decoder(tgt, encoder_output_key, encoder_output_value, src_mask, tgt_mask, layer_caches)
        # Returing the target embeddings, the output of encoder, and both source and target masks
        # The target mask ensures that the model won't see future elements of the sequence
        return output , new_caches

    #Applying projection layer with the Softmax Function to the decoder output
    def project(self, x):
        if self.projection_layer is None:
            raise ValueError("Projection layer is not initialized.")
        return self.projection_layer(x)
