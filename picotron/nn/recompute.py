"""
Selective Activation Recomputation (gradient checkpointing) utilities.
Allows checkpointing only memory-heavy components (like MLP or Attention blocks) selectively.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def checkpoint_wrapper(module: nn.Module, enabled: bool = True):
    """
    Wrap module to run forward passes under torch.utils.checkpoint.checkpoint.
    """
    if not enabled:
        return module
        
    class CheckpointedModule(nn.Module):
        def __init__(self, raw_module: nn.Module):
            super().__init__()
            self.raw_module = raw_module
            
        def forward(self, *args, **kwargs):
            # Checkpoint forward pass execution to recompute activations during backward pass
            # We use use_reentrant=False for compatibility with mixed precision
            return checkpoint(self.raw_module, *args, use_reentrant=False, **kwargs)
            
    return CheckpointedModule(module)
