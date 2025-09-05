# Universal Transformers

A PyTorch implementation of Universal Transformers - a neural sequence model that shares weights across depth using a recurrent inductive bias. Now enhanced with Adaptive Computation Time (ACT) for dynamic computation allocation.

## Overview

Universal Transformers combine the parallelizability of Transformers with the recurrent depth processing of RNNs. Instead of having separate layers, they use a single shared block that is applied recurrently across depth, with depth/time embeddings to distinguish different processing steps.

This implementation includes both the original Universal Transformer and an enhanced version with **Adaptive Computation Time (ACT)**, which allows the model to dynamically decide how many computation steps to perform at each position, optimizing both performance and computational efficiency.

## Key Features

- **Recurrent Depth Processing**: Single shared decoder block applied T times with depth embeddings
- **Adaptive Computation Time (ACT)**: Dynamic computation allocation with learned halting probabilities
- **Efficient Inference**: Grouped Query Attention (GQA) for faster autoregressive generation
- **KV Caching**: Optimized for sequential decoding with key-value cache
- **Flexible Sampling**: Support for top-k and top-p (nucleus) sampling
- **Pre-norm Architecture**: RMS normalization with residual connections
- **Ponder Cost Regularization**: Trainable trade-off between performance and computational cost

## Architecture

### Standard Universal Transformer
```
Input → Positional Embeddings → [Depth/Time Embeddings + Shared Decoder Block] × T → Output
```

### Universal Transformer with ACT
```
Input → Positional Embeddings → [Depth/Time Embeddings + Shared Decoder Block] × T_adaptive → ACT Weighting → Output
```

Where the shared decoder block consists of:
- RMSNorm → Masked Self-Attention → Residual → RMSNorm → Feed-Forward → Residual

The ACT mechanism adds:
- **Halting Head**: Computes position-wise halting probabilities p_i^(t) = σ(W_h · h_i^(t) + b_h)
- **Adaptive Stopping**: Each position halts when cumulative probability ≥ threshold (1-ε)
- **Weighted Output**: Final states are weighted combinations of all computation steps

## Project Structure
```
Universal_Transformers/
├── __init__.py                     # Package initialization
├── ut_decoder.py                   # Main Universal Decoder class
├── ut_decoder_bloc.py              # Shared decoder block
├── ut_block.py                     # Universal Transformer block wrapper
├── ut.py                          # Complete Universal Transformer model
├── transition_mlp.py              # Feed-forward transition layer
├── depth_time_emdedding.py        # Depth/time position embeddings
├── pos_emb.py                     # Sinusoidal positional embeddings
├── train_ut_decoder.py            # Training script for the model
├── UT_ACT/                        # Adaptive Computation Time implementation
│   ├── __init__.py                # ACT package initialization
│   ├── act_module.py              # Core ACT mechanism and halting head
│   ├── ut_decoder_with_act.py     # UT decoder enhanced with ACT
│   ├── train_ut_act.py            # Training script for UT+ACT
│   └── dummy_dataset.py           # Dataset utilities for ACT training
└── transformers/                  # Standard transformer components
    ├── gqa.py                     # Grouped Query Attention
    ├── rms_norm.py                # RMS normalization
    └── ...                        # Other transformer utilities
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Universal_Transformers.git
cd Universal_Transformers

# Install dependencies
pip install torch numpy tqdm
```

## Quick Start

### Standard Universal Transformer

```python
import torch
from Universal_Transformers.ut_decoder import UniversalTransformerDecoder

# Model configuration
config = {
    'vocab_size': 30000,
    'dim': 512,
    'max_seq_len': 1024,
    'num_heads': 8,
    'num_kv_heads': 2,  # For GQA
    'T': 6,  # Number of recurrent steps
    'dropout': 0.1
}

# Initialize model
model = UniversalTransformerDecoder(**config)

# Forward pass
input_ids = torch.randint(0, config['vocab_size'], (1, 50))
output = model(input_ids)

# Generate text
generated = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

### Universal Transformer with ACT

```python
import torch
from Universal_Transformers.UT_ACT.ut_decoder_with_act import UniversalDecoderWithACT

# ACT-enhanced model configuration
act_config = {
    'vocab_size': 30000,
    'dim': 512,
    'max_seq_len': 1024,
    'num_heads': 8,
    'num_kv_heads': 2,
    'max_steps': 12,  # Maximum computation steps
    'dropout': 0.1,
    'act_threshold': 0.99,  # Halt when cumulative prob ≥ 0.99
    'ponder_cost_weight': 0.01  # τ parameter for ponder cost
}

# Initialize ACT model
act_model = UniversalDecoderWithACT(**act_config)

# Forward pass with adaptive computation
input_ids = torch.randint(0, act_config['vocab_size'], (1, 50))
logits, ponder_costs, act_info = act_model(input_ids, return_act_info=True)

# Generate with ACT statistics
generated, act_stats = act_model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8,
    verbose_act=True  # Get per-token computation statistics
)

# Analyze adaptive computation
if act_stats:
    for i, stat in enumerate(act_stats[:5]):
        print(f"Token {i}: {stat['avg_computation_steps']:.2f} UT steps")
```

## Adaptive Computation Time (ACT)

The ACT mechanism allows the model to adaptively allocate computation based on input complexity:

### Key Benefits
- **Dynamic Computation**: Each position can use different amounts of computation
- **Efficiency**: Easy tokens require fewer steps, hard tokens get more computation
- **Learnable Trade-offs**: Ponder cost parameter (τ) balances accuracy vs efficiency
- **Interpretability**: Computation patterns reveal model's "thinking" process

### ACT Mathematics
The ACT mechanism implements the following mathematical framework:

1. **Halting Probabilities**: `p_i^(t) = σ(W_h · h_i^(t) + b_h)`
2. **Cumulative Probability**: `N_i^(t) = Σ p_i^(k)` for k=1 to t
3. **Halting Condition**: Stop when `N_i^(t) ≥ 1-ε` (threshold)
4. **Final Output**: `ẽ_i = Σ p_i^(t) * h_i^(t) + R_i * h_i^(T_i)`
5. **Ponder Cost**: `Ω = τ * mean(computation_steps_per_position)`

### Training with ACT
The total loss combines task performance with computational efficiency:
```
L_total = L_task + τ * Ω
```

Where:
- `L_task` is the standard language modeling loss
- `Ω` is the ponder cost (average computation time)
- `τ` controls the trade-off between accuracy and efficiency

## Training

### Standard Universal Transformer

To train the standard model, use the provided training script:

```bash
python train_ut_decoder.py
```

### Universal Transformer with ACT

To train with Adaptive Computation Time:

```bash
python Universal_Transformers_/UT_ACT/train_ut_act.py
```

### Training Configuration

Both training scripts support extensive configuration:

**Standard UT Training:**
- **Model Config**: Adjust `vocab_size`, `dim`, `num_heads`, `T` (recurrent steps), etc.
- **Training Config**: Set `batch_size`, `learning_rate`, `num_epochs`, `warmup_steps`, etc.

**ACT Training Additional Parameters:**
- **`act_threshold`**: Halting threshold (default: 0.99) - higher values require more computation
- **`ponder_cost_weight`** (τ): Trade-off between accuracy and efficiency (default: 0.01)
- **`max_steps`**: Maximum computation steps allowed (default: 12)

The ACT trainer includes specialized logging for:
- Task loss (standard language modeling)
- Ponder cost (computational efficiency)
- Average computation steps per position
- Halting pattern analysis

Both trainers support logging, checkpoints, validation, and gradient clipping. They use dummy datasets by default for testing; replace with real data for production training.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- tqdm (for progress bars)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
