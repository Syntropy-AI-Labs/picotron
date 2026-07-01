"""
Expose all fused Triton kernels and their PyTorch fallbacks.
"""

from picotron.kernels.triton_kernels.core import (
    triton_rmsnorm,
    triton_layernorm,
    triton_swiglu,
    triton_geglu,
    triton_cross_entropy
)

from picotron.kernels.triton_kernels.attention import (
    triton_rope,
    triton_softmax,
    triton_dropout
)

from picotron.kernels.triton_kernels.blocks import (
    triton_residual_add_rmsnorm,
    triton_bias_activation
)

from picotron.kernels.triton_kernels.optimizers import (
    triton_adamw,
    triton_sgd
)
