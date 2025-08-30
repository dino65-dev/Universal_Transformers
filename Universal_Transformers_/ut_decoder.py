import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers_.gqa import GroupedQueryAttention
from .transition_mlp import FeedForwardTransition
from .transformers_.rms_norm import RMSNorm
from typing import Optional, Tuple, Dict
from .depth_time_emdedding import DepthTimeEmbedding
from .ut_decoder_bloc import UniversalTransformersDecoderBlock
from .pos_emb import _create_sinusoidal_embeddings
import math
class UniversalTransformerDecoder(nn.Module):
    """
    Decoder-only Universal Transformer for autoregressive language modeling.
    Uses recurrent depth with shared weights across T steps.
    """
    def __init__(self, vocab_size: int, dim: int = 512, max_seq_len: int = 1024, num_heads: int = 8, num_kv_heads: int = 8, T: int = 6, dff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.T = T
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size,dim)

        # Positional encoding (standard sinusoidal, applied once)
        self.register_buffer("pos_emb", _create_sinusoidal_embeddings(max_seq_len, dim))

        # Depth/time embedding ()
        self.depth_embed = DepthTimeEmbedding(dim)

        # Shared UT block (used T times)
        self.shared_block = UniversalTransformersDecoderBlock(dim,num_heads,num_kv_heads,dropout=dropout,d_ff=dff)

        # Final processing
        self.final_norm = RMSNorm(dim=dim)
        self.lm_head = nn.Linear(dim,vocab_size,bias=False)

        # Precompute Causal maks
        self.register_buffer("causal_mask",torch.tril(torch.ones(max_seq_len,max_seq_len,dtype=torch.bool)))

    def forward(self, input_ids: torch.Tensor, cache: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,use_cache: bool = False):
        """
        input_ids: [batch, seq_len] token indices
        cache: Optional KV cache for autoregressive decoding
            Format: {'step_{t}': (past_k, past_v)} for each UT step t
        use_cache: Whether to return cache for next forward pass

        Returns:
            logits: [batch, seq_len, vocab_size]
            new_cache: Updated cache if use_cache=True

        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings + pos emb (applied once)
        x = self.token_embedding(input_ids)
        x = x + self.pos_emb[:,:seq_len,:] # Add pos encoding

        # Prepare causal mask for this seq len
        causal_mask = self.causal_mask[:seq_len,:seq_len] if seq_len <= self.max_seq_len else None

        #Initialize cache tracking
        new_cache = {}
        current_cache = cache if cache is not None else {}

        # Recurrent loop over T depth steps
        for t in range(1, self.T + 1): # t from 1 to T (1-indexed like your original code)
            #Add depth/time embedding for this step
            x = self.depth_embed(x,t)

            # Get cache for this specific UT step
            step_cache = None
            if f'step_{t}' in current_cache:
                step_cache = current_cache[f'step_{t}']

            # Apply shared UT decoder block
            x, step_new_cache = self.shared_block(x,causal_mask,step_cache)

            # store cache for this step
            if use_cache:
                new_cache[f'step_{t}'] = step_new_cache

        #Final normalization and projection to vocab
        x = self.final_norm(x)
        logits = self.lm_head(x) # [B,S, vocab_size]

        return logits, new_cache

    def generate(self,input_ids: torch.Tensor,max_new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None, do_sample: bool = True):
        """
        Autoregressive generation using the UT decoder.

        Args:
            input_ids: [batch, seq_len] initial tokens
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling
            do_sample: Whether to sample or use greedy

        Returns:
            generated_ids: [batch, seq_len + max_new_tokens]
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        generated = input_ids
        kv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        for _ in range(max_new_tokens):
            # Forward pass (only compute last token when using cache)
            if len(kv_cache) > 0:
                # During generation, only need to process the last token
                input_for_forward = generated[:, -1:]
            else:
                input_for_forward = generated

            with torch.no_grad():
                logits, updated_cache = self.forward(input_for_forward, cache=kv_cache, use_cache=True)
                if updated_cache is not None:
                    kv_cache = updated_cache

            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / temperature  # [batch, vocab_size]

            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check if we've exceeded max_seq_len
            if generated.shape[1] >= self.max_seq_len:
                break

        return generated
