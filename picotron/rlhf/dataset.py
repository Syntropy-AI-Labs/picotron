"""
Shared preference dataset loader class for DPO, ORPO, and GRPO.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict

class PreferenceDataset(Dataset):
    """
    Standard preference dataset containing prompt-chosen and prompt-rejected token IDs.
    """
    def __init__(self, data: List[Dict[str, List[int]]], max_length: int = 512):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _pad_or_truncate(self, tokens: List[int], pad_val: int = 0) -> torch.Tensor:
        if len(tokens) > self.max_length:
            return torch.tensor(tokens[:self.max_length], dtype=torch.long)
        # Pad sequence
        padded = tokens + [pad_val] * (self.max_length - len(tokens))
        return torch.tensor(padded, dtype=torch.long)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item.get("prompt", [])
        chosen = item.get("chosen", [])
        rejected = item.get("rejected", [])
        
        # Build chosen sequence and rejected sequence
        chosen_seq = prompt + chosen
        rejected_seq = prompt + rejected
        
        # Build labels (mask out prompt tokens with -100)
        chosen_labels = [-100] * len(prompt) + chosen
        rejected_labels = [-100] * len(prompt) + rejected
        
        return {
            "chosen_input": self._pad_or_truncate(chosen_seq, pad_val=0),
            "chosen_labels": self._pad_or_truncate(chosen_labels, pad_val=-100),
            "rejected_input": self._pad_or_truncate(rejected_seq, pad_val=0),
            "rejected_labels": self._pad_or_truncate(rejected_labels, pad_val=-100),
        }
