import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

class ACTHaltingHead(nn.Module):
    """
    Computes halting probabilities for each position at each step.

    From the math: p_i^(t) = σ(W_h · h_i^(t) + b_h)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        returns: [batch, seq_len] halting probabilities in (0,1)
        """
        logits = self.linear(x).squeeze(-1)  # [batch, seq_len]
        return torch.sigmoid(logits)

class AdaptiveComputationTime(nn.Module):
    """
    Full ACT mechanism implementing the mathematical framework:

    1. Compute halting probabilities p_i^(t) at each step
    2. Accumulate N_i^(t) = Σ p_i^(k) for k=1 to t
    3. Halt when N_i^(t) >= 1-ε
    4. Final output: ẽ_i = Σ p_i^(t) * h_i^(t) + R_i * h_i^(T_i)
    """
    def __init__(
        self,
        d_model: int,
        threshold: float = 0.99,  # 1-ε from the math
        max_steps: int = 12,
        ponder_cost_weight: float = 0.01
    ):
        super().__init__()
        self.halting_head = ACTHaltingHead(d_model)
        self.threshold = threshold
        self.max_steps = max_steps
        self.ponder_cost_weight = ponder_cost_weight

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Apply ACT to a sequence of hidden states from UT recurrence steps.

        Args:
            hidden_states: [batch, max_steps, seq_len, d_model]
                          States from each UT recurrence step

        Returns:
            final_output: [batch, seq_len, d_model] - weighted combination
            ponder_cost: [batch] - average pondering time per sample
            act_info: dict with detailed ACT statistics
        """
        batch_size, max_steps, seq_len, d_model = hidden_states.shape
        device = hidden_states.device

        # Initialize tracking tensors
        cumulative_prob = torch.zeros(batch_size, seq_len, device=device)
        final_output = torch.zeros(batch_size, seq_len, d_model, device=device)
        n_updates = torch.zeros(batch_size, seq_len, device=device)
        still_running = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Store per-step info for analysis
        step_probs = []
        step_halts = []

        for step in range(max_steps):
            if not still_running.any():
                break

            # Current hidden state for this step
            h_t = hidden_states[:, step, :, :]  # [batch, seq_len, d_model]

            # Compute halting probabilities: p_i^(t) = σ(W_h · h_i^(t) + b_h)
            p_t = self.halting_head(h_t)  # [batch, seq_len]

            # Only compute for positions still running
            p_t = p_t * still_running.float()

            # Check which positions should halt: N_i^(t) >= 1-ε
            new_cumulative = cumulative_prob + p_t
            should_halt = (new_cumulative >= self.threshold) & still_running

            # Calculate remainders for newly halting positions: R_i = 1 - Σ p_i^(k)
            remainders = torch.where(
                should_halt,
                1.0 - cumulative_prob,  # R_i = 1 - N_i^(t-1)
                p_t  # Still running positions use full probability
            )

            # Accumulate weighted states: ẽ_i += weight * h_i^(t)
            final_output = final_output + remainders.unsqueeze(-1) * h_t

            # Update tracking variables
            cumulative_prob = new_cumulative
            n_updates = n_updates + still_running.float()

            # Update mask for next iteration
            still_running = still_running & (~should_halt)

            # Store step info
            step_probs.append(p_t.detach())
            step_halts.append(should_halt.detach())

        # Handle positions that never halted (edge case)
        never_halted = still_running
        if never_halted.any():
            remaining_weight = 1.0 - cumulative_prob
            final_output = final_output + (remaining_weight.unsqueeze(-1) *
                                         hidden_states[:, -1, :, :])
            n_updates = n_updates + never_halted.float()

        # Compute ponder cost: Ω = Σ (T_i + R_i)
        effective_steps = n_updates + (1.0 - cumulative_prob).clamp(min=0.0)
        ponder_cost = effective_steps.mean(dim=1)  # Average over sequence length

        # Detailed info for analysis/debugging
        act_info = {
            'n_updates': n_updates,
            'effective_steps': effective_steps,
            'step_probs': torch.stack(step_probs, dim=0) if step_probs else None,
            'step_halts': torch.stack(step_halts, dim=0) if step_halts else None,
            'final_cumulative': cumulative_prob
        }

        return final_output, ponder_cost, act_info
