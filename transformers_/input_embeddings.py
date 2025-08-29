import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_mdoel: int, vocab_size : int) -> None:
        super().__init__()
        self.d_model = d_mdoel #dimension of vectors
        self.vocab_size = vocab_size # total unique words
        self.embedding = nn.Embedding(vocab_size, d_mdoel) #pytorch layer that converts integer indices to dense embeddings

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #Normalizing the variance of the embedding
