import torch.nn as nn
from layer_normalization import LayerNormalization
class ResidualConnection(nn.Module):
    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer): # sublayer can be attention or FFN
    # this is pre_norm but in actual paper it was post_norm
    # pre_norm  has adv of stability and fast convergence
        return x + self.dropout(sublayer(self.norm(x)))
