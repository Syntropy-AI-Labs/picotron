"""
Triton optimizers: Fused AdamW, AdamW 8-bit, and SGD kernels.
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
# 1. FUSED ADAMW KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _adamw_kernel(
        Param_ptr, Grad_ptr, M_ptr, V_ptr,
        lr, beta1, beta2, eps, wd, step,
        N,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        p = tl.load(Param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(Grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(M_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        p = p * (1.0 - lr * wd)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        
        tl.store(M_ptr + offsets, m, mask=mask)
        tl.store(V_ptr + offsets, v, mask=mask)
        
        bias_corr1 = 1.0 - tl.math.pow(beta1, step)
        bias_corr2 = 1.0 - tl.math.pow(beta2, step)
        
        step_sz = lr * tl.math.sqrt(bias_corr2) / bias_corr1
        p_new = p - step_sz * m / (tl.math.sqrt(v) + eps)
        tl.store(Param_ptr + offsets, p_new.to(Param_ptr.dtype.element_ty), mask=mask)

def triton_adamw(params, grads, exp_avgs, exp_avg_sqs, lr, beta1, beta2, eps, wd, step):
    if _TRITON_AVAILABLE and params.is_cuda:
        N = params.numel()
        block_size = 1024
        grid = (math.ceil(N / block_size),)
        _adamw_kernel[grid](
            params, grads, exp_avgs, exp_avg_sqs,
            lr, beta1, beta2, eps, wd, step,
            N, BLOCK_SIZE=block_size
        )
        return
        
    params.mul_(1 - lr * wd)
    exp_avgs.mul_(beta1).add_(grads, alpha=1 - beta1)
    exp_avg_sqs.mul_(beta2).addcmul_(grads, grads, value=1 - beta2)
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    denom = exp_avg_sqs.sqrt().add_(eps)
    params.addcdiv_(exp_avgs, denom, value=-step_size)

# =====================================================================
# 2. TRITON SGD KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _sgd_kernel(
        Param_ptr, Grad_ptr, Momentum_ptr,
        lr, momentum_factor, dampening, weight_decay, nestrov,
        N,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        p = tl.load(Param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(Grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Apply weight decay
        g = g + weight_decay * p
        
        # Apply momentum updates
        buf = tl.load(Momentum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        buf = momentum_factor * buf + (1.0 - dampening) * g
        tl.store(Momentum_ptr + offsets, buf, mask=mask)
        
        if nestrov:
            g = g + momentum_factor * buf
        else:
            g = buf
            
        p_new = p - lr * g
        tl.store(Param_ptr + offsets, p_new.to(Param_ptr.dtype.element_ty), mask=mask)

def triton_sgd(params, grads, momentum_buffers, lr, momentum_factor, dampening, weight_decay, nestrov):
    if _TRITON_AVAILABLE and params.is_cuda:
        N = params.numel()
        block_size = 1024
        grid = (math.ceil(N / block_size),)
        _sgd_kernel[grid](
            params, grads, momentum_buffers,
            lr, momentum_factor, dampening, weight_decay, nestrov,
            N, BLOCK_SIZE=block_size
        )
        return
        
    # Standard PyTorch SGD implementation
    d_p = grads
    if weight_decay != 0:
        d_p = d_p.add(params, alpha=weight_decay)
    if momentum_factor != 0:
        buf = momentum_buffers
        buf.mul_(momentum_factor).add_(d_p, alpha=1 - dampening)
        if nestrov:
            d_p = d_p.add(buf, alpha=momentum_factor)
        else:
            d_p = buf
    params.add_(d_p, alpha=-lr)
