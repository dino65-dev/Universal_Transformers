# Universal Transformers

A PyTorch implementation of Universal Transformers - a neural sequence model that shares weights across depth using a recurrent inductive bias.

## Overview

Universal Transformers combine the parallelizability of Transformers with the recurrent depth processing of RNNs. Instead of having separate layers, they use a single shared block that is applied recurrently across depth, with depth/time embeddings to distinguish different processing steps.

## Key Features

- **Recurrent Depth Processing**: Single shared decoder block applied T times with depth embeddings
- **Efficient Inference**: Grouped Query Attention (GQA) for faster autoregressive generation
- **KV Caching**: Optimized for sequential decoding with key-value cache
- **Flexible Sampling**: Support for top-k and top-p (nucleus) sampling
- **Pre-norm Architecture**: RMS normalization with residual connections

## Architecture

```
Input → Positional Embeddings → [Depth/Time Embeddings + Shared Decoder Block] × T → Output
```

Where the shared decoder block consists of:
- RMSNorm → Masked Self-Attention → Residual → RMSNorm → Feed-Forward → Residual

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
└── transformers_/                 # Standard transformer components
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
pip install torch numpy
```

## Quick Start

```python
import torch
from Universal_Transformers import UniversalDecoder

# Model configuration
config = {
    'vocab_size': 30000,
    'd_model': 512,
    'num_heads': 8,
    'num_kv_heads': 2,  # For GQA
    'd_ff': 2048,
    'max_seq_len': 1024,
    'depth': 6,  # Number of recurrent steps
    'dropout': 0.1
}

# Initialize model
model = UniversalDecoder(**config)
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
