"""
Context Parallel (CP) utilities to shard activations along the temporal/sequence dimension.
"""

import torch
import torch.distributed as dist
from typing import List, Optional

def shard_context(tensor: torch.Tensor, cp_size: int, cp_rank: int, dim: int = 1) -> torch.Tensor:
    """
    Shard a sequence tensor along the configured dimension (default is seq_len dimension 1) for CP.
    """
    if cp_size <= 1:
        return tensor
        
    seq_len = tensor.size(dim)
    chunk_size = seq_len // cp_size
    return tensor.narrow(dim, cp_rank * chunk_size, chunk_size)

def gather_context(tensor: torch.Tensor, cp_group: Optional[dist.ProcessGroup] = None, dim: int = 1) -> torch.Tensor:
    """
    Gather sharded sequence chunks from all CP ranks.
    """
    if cp_group is None or not dist.is_initialized():
        return tensor
        
    cp_size = dist.get_world_size(cp_group)
    if cp_size <= 1:
        return tensor
        
    tensor_list = [torch.empty_like(tensor) for _ in range(cp_size)]
    dist.all_gather(tensor_list, tensor, group=cp_group)
    return torch.cat(tensor_list, dim=dim)
