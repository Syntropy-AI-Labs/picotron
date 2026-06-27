"""
Pure PyTorch implementation of Root Mean Square Normalization (RMSNorm).
Avoids external dependencies like Triton or CUDA extensions.
"""

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization layer.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """Initialize RMSNorm parameters."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the input tensor."""
        # Calculate variance using manual formula for precision control
        variance = x.pow(2).mean(-1, keepdim=True)
        # Compute scaling factor
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed
