"""
Triton block connections: Fused Residual Add + Normalization, and Fused Bias + Activation.
Includes fallbacks to standard PyTorch implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton availability flag
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass

# =====================================================================
# 1. FUSED RESIDUAL ADD + RMSNORM KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _residual_add_rmsnorm_fwd_kernel(
        X_ptr, Residual_ptr, Y_ptr, W_ptr, Rstd_ptr,
        stride_r, N_cols, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        X_ptr += row_idx * stride_r
        Residual_ptr += row_idx * stride_r
        Y_ptr += row_idx * stride_r
        
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_cols
        
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(Residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Fused residual addition
        x_new = x + res
        tl.store(X_ptr + offsets, x_new.to(X_ptr.dtype.element_ty), mask=mask)
        
        # Norm calculation on newly fused residual vector
        var = tl.sum(x_new * x_new, axis=0) / N_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd_ptr + row_idx, rstd)
        
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x_new * rstd * w
        tl.store(Y_ptr + offsets, y.to(Y_ptr.dtype.element_ty), mask=mask)

def triton_residual_add_rmsnorm(x, residual, weight, eps=1e-5):
    """Fuses adding residual link and computing RMSNorm on the output in one pass."""
    if _TRITON_AVAILABLE and x.is_cuda:
        M, N = x.shape[0] * x.shape[1], x.shape[2]
        x_flat = x.view(M, N)
        res_flat = residual.view(M, N)
        y = torch.empty_like(x_flat)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        block_size = triton.next_power_of_2(N)
        _residual_add_rmsnorm_fwd_kernel[(M,)](
            x_flat, res_flat, y, weight, rstd,
            N, N, eps,
            BLOCK_SIZE=block_size
        )
        return y.view_as(x)
        
    # PyTorch fallback
    x_new = x + residual
    variance = x_new.pow(2).mean(-1, keepdim=True)
    return x_new * torch.rsqrt(variance + eps) * weight

# =====================================================================
# 2. FUSED BIAS + ACTIVATION KERNEL (SiLU/GELU)
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _bias_activation_fwd_kernel(
        X_ptr, Bias_ptr, Out_ptr,
        N_cols, N_elements, type_flag,  # 0 for SiLU, 1 for GELU
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_elements
        
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Load associated linear bias coordinate
        col_idx = offsets % N_cols
        bias = tl.load(Bias_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
        
        # Fused bias addition
        x_biased = x + bias
        
        if type_flag == 0:
            out = x_biased * tl.sigmoid(x_biased)
        else:
            out = x_biased * 0.5 * (1.0 + tl.math.erf(x_biased * 0.70710678))
            
        tl.store(Out_ptr + offsets, out.to(Out_ptr.dtype.element_ty), mask=mask)

def triton_bias_activation(x, bias, activation_type="silu"):
    """Fuses linear bias addition and activation computation."""
    if _TRITON_AVAILABLE and x.is_cuda:
        N_elements = x.numel()
        N_cols = x.shape[-1]
        out = torch.empty_like(x)
        block_size = 1024
        grid = (math.ceil(N_elements / block_size),)
        
        type_flag = 1 if activation_type == "gelu" else 0
        _bias_activation_fwd_kernel[grid](
            x, bias, out,
            N_cols, N_elements, type_flag,
            BLOCK_SIZE=block_size
        )
        return out
        
    # PyTorch fallback
    x_biased = x + bias
    return F.gelu(x_biased) if activation_type == "gelu" else F.silu(x_biased)
