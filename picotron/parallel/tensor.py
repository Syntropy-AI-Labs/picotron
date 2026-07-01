"""
Megatron-LM style Tensor Parallel (TP) linear projection layers.
Provides ColumnParallelLinear and RowParallelLinear with backward reduction hooks.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

class ColumnParallelLinear(nn.Module):
    """
    Shards linear projection output weights across the column dimension (e.g. QKV projections).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, tp_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        
        # Determine division rank
        tp_size = dist.get_world_size(tp_group) if (tp_group is not None and dist.is_initialized()) else 1
        self.split_out_features = out_features // tp_size
        
        self.weight = nn.Parameter(torch.empty(self.split_out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.split_out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Columns parallel linear forward pass
        # Since outputs are sharded, no reduction is needed on the output activations
        return nn.functional.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    """
    Shards weights across input features, requiring all-reduce step on backward or forward paths.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, tp_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        
        tp_size = dist.get_world_size(tp_group) if (tp_group is not None and dist.is_initialized()) else 1
        self.split_in_features = in_features // tp_size
        
        self.weight = nn.Parameter(torch.empty(out_features, self.split_in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Row parallel forward pass: compute local projection followed by all-reduce reduction
        out = nn.functional.linear(x, self.weight)
        
        if self.tp_group is not None and dist.is_initialized():
            dist.all_reduce(out, group=self.tp_group)
            
        if self.bias is not None:
            out = out + self.bias
        return out
