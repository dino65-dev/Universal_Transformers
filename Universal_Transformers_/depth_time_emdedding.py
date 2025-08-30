import torch
import torch.nn as nn
import math

class DepthTimeEmbedding(nn.Module):
    """
    Sinusoidal depth/timestep embedding for UT depth recurrence.
    Adds a depth-dependent sinusoidal vector to every position at step t (1-indexed).
    """
    def __init__(self,d_model : int , base : float = 10000.0) :
        super().__init__()
        self.d_model = d_model
        self.base = base

        # Precompute the frequency denominators for speed
        inv_freq = torch.exp(-math.log(base) * torch.arange(0,d_model,2).float() / d_model )

        #register_buffer("inv_freq",self.inv_freq,False)
        #shape [d_model/2]
        self.register_buffer("inv_freq",inv_freq,False)

    def forward(self,x: torch.Tensor , t: int) :
        """
        x: [batch, seq_len, d_model]
        t: current depth step (1, 2, ..., num_steps)
        returns x + depth_embed(t)
        """
        bsz , seqlen , d = x.shape
        device = x.device
        #depth scalar -> sinusodial vector of the size d_model
        # even index: sin(t * inv_freq) , odd idx: cos(t * inv_freq)
        t_val = torch.tensor(float(t),device=device)
        angles = t_val * self.inv_freq
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        depth_vec = torch.zeros(d,device=device)
        depth_vec[0::2] = sin
        depth_vec[1::2] = cos

        #broadcast to [bsz,seqlen, d_model]
        return x + depth_vec.view(1,1,d).expand(bsz,seqlen,d)


