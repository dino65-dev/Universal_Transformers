import torch.nn as nn
import torch

class LayerNormalization(nn.Module):

    def __init__(self, eps : float = 10**-6) -> None:
        self.eps = eps # keeping epsilon to prevent div with zero

        #creating gamma as a trainable parameter
        self.gamma = nn.Parameter(torch.ones(1))

        #creating beta or bias as trainable parameter
        self.beta = nn.Parameter(torch.ones(1))

    def forwrad(self,x):
        # dim = -1 means row wise
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.gamma * ((x - mean)/ (std + self.eps)) + self.beta
