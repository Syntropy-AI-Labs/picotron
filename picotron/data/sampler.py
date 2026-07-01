"""
Length-bucketed and dynamic batching samplers.
Groups sequences of similar lengths into batches to minimize padding tokens.
"""

import random
from torch.utils.data import Sampler
from typing import List, Iterator

class LengthBucketedSampler(Sampler[List[int]]):
    """
    Groups sequences of similar lengths into batches.
    """
    def __init__(self, sequence_lengths: List[int], batch_size: int, shuffle: bool = True):
        self.sequence_lengths = sequence_lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Sort indices by length
        self.sorted_indices = sorted(range(len(self.sequence_lengths)), key=lambda idx: self.sequence_lengths[idx])
        
    def __iter__(self) -> Iterator[List[int]]:
        # Group sorted indices into batches of size batch_size
        batches = [
            self.sorted_indices[i : i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        
        # Shuffle batches to prevent systemic ordering bias during training
        if self.shuffle:
            random.shuffle(batches)
            
        for batch in batches:
            yield batch
            
    def __len__(self) -> int:
        return (len(self.sequence_lengths) + self.batch_size - 1) // self.batch_size
