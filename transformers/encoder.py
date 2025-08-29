import torch.nn as nn
import torch
from layer_normalization import LayerNormalization
class Encoder(nn.Module):
    """An Encoder can have several Encoder Blocks"""

    def __init__(self,layers: nn.ModuleList) -> None:
        self.layers = layers # storing the EncoderBlocks
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        #Iterating over each EncoderBlock stored in self.layers
        for layer in self.layers:
            x = layer(x,mask) # Applying each EncoderBlock to the input tensor 'x'
        return self.norm(x) # normalizing after encoder operation, it's not in paper but in now a days it done for better training and stbility
    
