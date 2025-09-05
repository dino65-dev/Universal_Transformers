# dummy_dataset.py
import torch
from torch.utils.data import Dataset
from typing import Tuple

class DummyDataset(Dataset):
    """
    Dummy dataset for language modeling that generates random sequences.
    Each sample is a sequence of random tokens where target is input shifted by 1.
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random sequence of tokens
        # seq_len + 1 to create input and target (shifted by 1)
        sequence = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)

        # Input: first seq_len tokens
        input_ids = sequence[:-1]

        # Target: last seq_len tokens (shifted by 1 for next-token prediction)
        target_ids = sequence[1:]

        return input_ids, target_ids

class TextDataset(Dataset):
    """
    Real text dataset for when you have actual tokenized text data.
    """
    def __init__(self, tokenized_text: list, seq_len: int):
        super().__init__()
        self.data = tokenized_text
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract a chunk of seq_len + 1 tokens
        chunk = self.data[idx:idx + self.seq_len + 1]

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, target_ids
