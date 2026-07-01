"""
Fully Sharded Data Parallel (FSDP) sharding wrappers for Picotron.
Supports parameter/gradient offloading and mixed precision sharding policies.
"""

import torch
import torch.nn as nn
from typing import Optional

# Check FSDP availability
_FSDP_AVAILABLE = False
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        CPUOffload,
        BackwardPrefetch,
        ShardingStrategy
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    _FSDP_AVAILABLE = True
except ImportError:
    pass

def wrap_fsdp(
    model: nn.Module,
    block_class: type,
    mixed_precision_dtype: torch.dtype = torch.float16,
    cpu_offload: bool = False,
    sharding_strategy: str = "FULL_SHARD"
) -> nn.Module:
    """
    Wrap LLaMAModel using PyTorch Fully Sharded Data Parallel (FSDP).
    """
    if not _FSDP_AVAILABLE:
        print("FSDP is not available. Returning base model.")
        return model
        
    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=mixed_precision_dtype,
        reduce_dtype=mixed_precision_dtype,
        buffer_dtype=mixed_precision_dtype
    )
    
    # CPU offload policy
    offload_policy = CPUOffload(offload_params=cpu_offload)
    
    # Sharding Strategy
    strategy = ShardingStrategy.FULL_SHARD
    if sharding_strategy == "SHARD_GRAD_OP":
        strategy = ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "NO_SHARD":
        strategy = ShardingStrategy.NO_SHARD
        
    # Auto wrap policy based on block layer classes (TransformerBlock / DeltaNet)
    auto_wrap = transformer_auto_wrap_policy(
        transformer_layer_cls={block_class}
    )
    
    print(f"Wrapping model with FSDP (strategy: {sharding_strategy}, CPU offload: {cpu_offload})...")
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        cpu_offload=offload_policy,
        sharding_strategy=strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None
    )
