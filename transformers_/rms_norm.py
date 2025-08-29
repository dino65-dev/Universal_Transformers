import torch , torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,dim : int = 512, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The learnable scaling parameter, with a size of the feature dimension
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate the reciprocal of the square root for efficiency
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Normalize and then scale
        return self.gamma * self._norm(x.float()).type_as(x)
