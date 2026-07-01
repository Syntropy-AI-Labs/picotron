"""
Pipeline Parallel (PP) activation and gradient handshakes between device partitions.
"""

import torch
import torch.distributed as dist
from typing import Optional, Union

def send_forward(tensor: torch.Tensor, next_rank: int, group: Optional[dist.ProcessGroup] = None):
    """Send activations forward to the next pipeline stage."""
    if dist.is_initialized():
        dist.send(tensor=tensor, dst=next_rank, group=group)

def recv_forward(tensor_shape: tuple, dtype: torch.dtype, prev_rank: int, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Receive activations from the previous pipeline stage."""
    tensor = torch.empty(tensor_shape, dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu")
    if dist.is_initialized():
        dist.recv(tensor=tensor, src=prev_rank, group=group)
    return tensor

def send_backward(tensor: torch.Tensor, prev_rank: int, group: Optional[dist.ProcessGroup] = None):
    """Send gradients backward to the previous pipeline stage."""
    if dist.is_initialized():
        dist.send(tensor=tensor, dst=prev_rank, group=group)

def recv_backward(tensor_shape: tuple, dtype: torch.dtype, next_rank: int, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Receive gradients from the next pipeline stage."""
    tensor = torch.empty(tensor_shape, dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu")
    if dist.is_initialized():
        dist.recv(tensor=tensor, src=next_rank, group=group)
    return tensor
