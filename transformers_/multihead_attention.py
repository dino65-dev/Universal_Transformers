import torch.nn as nn
import torch
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model : int, head: int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = head

        #we ensure the dim of model is divisible by head
        assert d_model % head == 0,"d_model isn't divisible by h"

        #d_k is the dimension of each attention head's key , query and value vectors
        self.d_k = d_model // head # // is for int return and / for float return

        #Defining weight matrices
        self.w_q = nn.Linear(d_model , d_model) #w_q
        self.w_k = nn.Linear(d_model,d_model) #w_k
        self.w_v = nn.Linear(d_model,d_model) # w_v
        self.w_o = nn.Linear(d_model,d_model) # w_o

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout : nn.Dropout):

        d_k = query.shape[-1] # the last dimension of q,k,v

        # we calculate the Attention(q,k,v) (@ is for matrix multiplication)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        # Before applying the softmax, we apply the mask
        if mask is not None: #if mask is defined
            attention_scores.masked_fill_(mask == 0, -1e9) #if mask is == 0 replace with -1e9(inifinity)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None :
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self,q,k,v,mask,cache_kv = None):

        query = self.w_q(q) # Q matrix
        key = self.w_k(k) # K matrix
        value = self.w_v(v) # v matrix

        # Splitting results into smaller matrices for the different heads
        # Splitting embeddings (third dimension) into h parts
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension

        #implementing kv cache
        if cache_kv is not None:
            past_key , past_value = cache_kv
            key = torch.cat((past_key,key),dim=2)
            value = torch.cat((past_value,value),dim=2)
        present_kv = (key,value)

        #obtaining the output and the attention scores
        x, self.attention_scores = MultiHeadAttention.attention(query,key,value,mask,self.dropout)

        #Obtaining the Head matrix
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)

        return self.w_o(x) , present_kv # mulptiply the head matrix with output_weight matrix
