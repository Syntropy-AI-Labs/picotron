"""
Memory-mapped numpy dataset for training.
Reads tokenized inputs from binary files without heavy libraries or memory overhead.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    """
    Memory-mapped tokenized dataset.
    """
    def __init__(self, bin_path: str, sequence_length: int):
        """Initialize numpy memmap of tokenized indices."""
        super().__init__()
        self.bin_path = bin_path
        self.sequence_length = sequence_length
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary token file not found at: {bin_path}")
            
        # Determine dtype by inspecting filesize or suffix if needed, default to uint16 (or int32/uint32)
        # NanoGPT format usually writes as uint16
        # Let's inspect size of the file to determine token capacity
        # Or read standard uint16 (up to 65535 vocab) or fallback to int32 if file size is odd or too large.
        # We try uint16 first.
        self.dtype = np.uint16
        file_size = os.path.getsize(bin_path)
        
        # Safe-check: if file size is odd, it's not uint16/int32. If it's a multiple of 4 but not 2...
        # Let's default to np.uint16, but if it ends with '_int32.bin' or if size / 2 is not integer, fall back to int32.
        if "int32" in bin_path or file_size % 2 != 0:
            self.dtype = np.int32
            
        self.bytes_per_token = np.dtype(self.dtype).itemsize
        self.total_tokens = file_size // self.bytes_per_token
        
        # Memory-map the binary data
        self.data = np.memmap(bin_path, dtype=self.dtype, mode="r")
        
        # We need (sequence_length + 1) tokens per sample to construct inputs and targets (labels = inputs shifted by 1)
        self.sample_stride = sequence_length
        self.num_samples = (self.total_tokens - 1) // self.sample_stride

        if self.num_samples <= 0:
            raise ValueError(
                f"Dataset size ({self.total_tokens} tokens) is too small "
                f"for sequence_length {sequence_length}."
            )

    def __len__(self) -> int:
        """Return total samples available in dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch input and shifted label tensors at the given index."""
        start_idx = idx * self.sample_stride
        end_idx = start_idx + self.sequence_length + 1
        
        chunk = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
        
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
