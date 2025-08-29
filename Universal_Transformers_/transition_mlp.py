# This is Transition Function in Universal Transformer paper
import torch.nn as nn
import torch.nn.functional as F
import torch
class FeedForwardTransition(nn.Module):
    """
    Position-wise transition function (shared across depth steps).
    This is the UT "thinking" step: a 2-layer MLP with ReLU and dropout.
    """
    def __init__(self,d_model : int, d_ff: int = 2048, dropout: float = 0.1, activation: str = "relu"  ) :
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        if activation.lower() == "relu": # paper uses relu by default
            self.act = F.relu
        elif activation.lower() == "gelu":
            self.act = F.gelu
        else:
            raise ValueError("Activation Must be 'relu' or 'gelu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))
