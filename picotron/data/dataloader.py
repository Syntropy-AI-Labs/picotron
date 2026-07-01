"""
DataLoader configuration for Picotron.
Supports distributed sampling for data-parallel training (DDP).
"""

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing import Optional

def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    shuffle: bool = True,
) -> DataLoader:
    """
    Construct a PyTorch DataLoader with optional DistributedSampler for DDP.
    """
    sampler: Optional[DistributedSampler] = None
    
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True
        )
        # shuffle must be False when sampler is specified
        shuffle_loader = False
    else:
        sampler = None
        shuffle_loader = shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_loader,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
