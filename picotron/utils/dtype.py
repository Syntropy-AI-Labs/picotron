"""
Dtype selection utilities for Picotron.
Detects GPU compute capability and selects the optimal precision (BF16 or FP16).
"""

import torch

def get_default_dtype() -> torch.dtype:
    """
    Detect the device capability and return the appropriate default dtype.
    Returns torch.float16 if Compute Capability < 8.0, otherwise torch.bfloat16.
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            return torch.float16
        else:
            return torch.bfloat16
    return torch.float32  # Fallback for CPU training/testing
