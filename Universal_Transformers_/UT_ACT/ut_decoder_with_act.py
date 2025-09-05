# ut_decoder_with_act.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from ut_decoder_bloc import UniversalTransformersDecoderBlock  # Your existing UT block
from act_module import AdaptiveComputationTime
from depth_time_emdedding import DepthTimeEmbedding
from transformers_.rms_norm import RMSNorm

class UniversalDecoderWithACT(nn.Module):
    """
    UT Decoder with Adaptive Computation Time.

    Instead of fixed T steps, each position adaptively halts when it has
    computed "enough" based on learned halting probabilities.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        max_seq_len: int = 1024,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        max_steps: int = 12,  # Maximum UT recurrence steps
        dropout: float = 0.1,
        act_threshold: float = 0.99,
        ponder_cost_weight: float = 0.01
    ):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        self.max_seq_len = max_seq_len
        self.ponder_cost_weight = ponder_cost_weight

        # Core components (same as before)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.register_buffer("pos_emb", self._create_sinusoidal_embeddings(max_seq_len, dim))
        self.depth_embed = DepthTimeEmbedding(dim)
        self.shared_block = UniversalTransformersDecoderBlock(dim, num_heads, num_kv_heads,dropout= dropout)

        # ACT components
        self.act = AdaptiveComputationTime(
            d_model=dim,
            threshold=act_threshold,
            max_steps=max_steps,
            ponder_cost_weight=ponder_cost_weight
        )

        # Output components
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        )

    def _create_sinusoidal_embeddings(self, max_len: int, dim: int) -> torch.Tensor:
        """Same as before"""
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        emb = torch.zeros(max_len, dim)
        emb[:, 0::2] = torch.sin(pos * div_term)
        emb[:, 1::2] = torch.cos(pos * div_term)

        return emb.unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
        return_act_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass with ACT.

        Returns:
            logits: [batch, seq_len, vocab_size]
            ponder_cost: [batch] if training, else None
            act_info: Dict with ACT statistics if return_act_info=True
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initial embeddings + positional encoding (applied once)
        x = self.token_embedding(input_ids)  # [B, S, D]
        x = x + self.pos_emb[:, :seq_len, :]

        # Prepare causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len] if seq_len <= self.max_seq_len else None

        # Store all intermediate states for ACT
        all_hidden_states = []

        # UT recurrence loop (collect all intermediate states)
        current_state = x
        for step in range(1, self.max_steps + 1):
            # Add depth embedding
            current_state = self.depth_embed(current_state, step)

            # Apply shared UT block
            current_state, _ = self.shared_block(current_state, causal_mask, cache=None)

            # Store this step's state
            all_hidden_states.append(current_state)

        # Stack all states: [batch, max_steps, seq_len, dim]
        hidden_states_tensor = torch.stack(all_hidden_states, dim=1)

        # Apply ACT to get final weighted output
        final_states, ponder_cost, act_info = self.act(hidden_states_tensor)

        # Final processing
        output = self.final_norm(final_states)
        logits = self.lm_head(output)

        return_values = [logits]

        if self.training or return_act_info:
            return_values.append(ponder_cost)
        else:
            return_values.append(None)

        if return_act_info:
            return_values.append(act_info)
        else:
            return_values.append(None)

        return tuple(return_values)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        verbose_act: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Generation with ACT - shows adaptive computation per token.
        """
        self.eval()
        device = input_ids.device

        generated = input_ids
        act_stats = [] if verbose_act else None

        for step in range(max_new_tokens):
            with torch.no_grad():
                logits, _, act_info = self.forward(
                    generated,
                    return_act_info=verbose_act
                )

            if verbose_act and act_info is not None:
                # Log ACT statistics for this generation step
                avg_steps = act_info['effective_steps'].mean().item()
                act_stats.append({
                    'generation_step': step,
                    'avg_computation_steps': avg_steps,
                    'per_position_steps': act_info['effective_steps'][0].tolist()
                })

            # Sample next token (same as before)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if generated.shape[1] >= self.max_seq_len:
                break

        return generated, act_stats
