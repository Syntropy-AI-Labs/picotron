"""
Expert Parallel (EP) communication utilities for sharding Mixture of Experts (MoE).
"""

import torch
import torch.distributed as dist
from typing import Optional

def dispatch_tokens_to_experts(
    tokens: torch.Tensor,
    dispatch_indices: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """
    All-to-All communication to route token representations to the corresponding expert ranks.
    """
    if ep_group is None or not dist.is_initialized():
        return tokens
        
    ep_size = dist.get_world_size(ep_group)
    if ep_size <= 1:
        return tokens
        
    # Standard PyTorch all-to-all distributed handoff
    output_tokens = torch.empty_like(tokens)
    dist.all_to_all_single(output_tokens, tokens, group=ep_group)
    return output_tokens
