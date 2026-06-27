"""
ZeRO-1 Optimizer wrapper using pure torch.distributed.
Partitions the optimizer states (e.g., Adam momentum and variance) across the DDP rank group.
"""

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import List, Dict, Any, Tuple, Optional

class ZeroRedundancyOptimizer(Optimizer):
    """
    ZeRO-Stage 1 style optimizer wrapper that partitions optimizer states across DP processes.
    """
    def __init__(
        self,
        params,
        optim_class,
        rank: int = 0,
        world_size: int = 1,
        process_group: Any = None,
        **defaults
    ):
        """Initialize the partition-based optimizer wrapper."""
        self.rank = rank
        self.world_size = world_size
        self.pg = process_group if process_group is not None else dist.group.WORLD
        
        # Flatten and keep reference to parameters
        self.all_params = list(params)
        
        # Partition parameters round-robin or contiguous. We do simple size-based partitioning.
        self.local_params = []
        for i, p in enumerate(self.all_params):
            if i % self.world_size == self.rank:
                self.local_params.append(p)
                
        # Initialize internal optimizer on local parameters only
        self.base_optimizer = optim_class(self.local_params, **defaults)
        
        # Super init
        super().__init__(self.all_params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Perform optimizer step, all-reducing gradients and updating local parameters."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Reduce gradients across DP group for all params
        for i, p in enumerate(self.all_params):
            if p.grad is None:
                continue
            
            # All-reduce gradients across group
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=self.pg)
            p.grad.mul_(1.0 / self.world_size)

        # 2. Local optimizer updates local states and parameters
        self.base_optimizer.step()

        # 3. Broadcast updated parameters from their host ranks to all ranks
        for i, p in enumerate(self.all_params):
            source_rank = i % self.world_size
            dist.broadcast(p, src=source_rank, group=self.pg)

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients for all parameters."""
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
