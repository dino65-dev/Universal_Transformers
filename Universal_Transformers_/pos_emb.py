# Positional Encoding
import torch
import math

def _create_sinusoidal_embeddings(max_len: int, dim: int):
    """Create sinusoidal positional embeddings"""
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

    emb = torch.zeros(max_len, dim)
    emb[:, 0::2] = torch.sin(pos * div_term)
    emb[:, 1::2] = torch.cos(pos * div_term)

    return emb.unsqueeze(0)  # [1, max_len, dim]
