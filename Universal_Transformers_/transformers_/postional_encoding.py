import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np

class PostionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_length : int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting

        # Creating a postional encoding matrix of shape(seq_len , d_model)
        self.pe = torch.zeros(self.seq_length, self.d_model)

        #Creating postion tensor of shape (1, seq_len) --> (1,1,seq_len)for batch handling
        position = torch.arange(0, self.seq_length,dtype=torch.float).unsqueeze(1)

        # Creating divison term of postional encoding using formula --> a^b = exp(b * log(a))
        div_term = torch.exp(torch.arange(0,self.d_model,2).float() * (-math.log(10000.0)/self.d_model))

        # Applying sine function for even indices
        self.pe[:,0::2] = torch.sin(position * div_term)

        #Applying cosine function for odd indices
        self.pe[:,1::2] = torch.cos(position * div_term)

        # for batch handling we increase pe matrix shape to (1,seq_len , d_model)
        self.pe = self.pe.unsqueeze(0)

        # Register pe matrix as buffer cause its not update gradient during backprop
        self.register_buffer('pe', self.pe)

    def forward(self,x):
        #Adding postional encoding to input data
        #Assume input x has shape [2, 3, 8] (batch_size=2, seq_len=3, d_model=8) cause seq_length can vary sentence to sentence
        x = x+ (self.pe[:, :x.shape[1],:]).requires_grad_(False) # not update gradients during backprop
        return self.dropout(x)

