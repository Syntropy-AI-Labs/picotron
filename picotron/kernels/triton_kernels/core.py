"""
Core Triton kernels: RMSNorm, LayerNorm, SwiGLU, GELU, and Cross Entropy.
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
# 1. FUSED RMSNORM KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr, Y_ptr, W_ptr, Rstd_ptr,
        stride_r, N_cols, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        X_ptr += row_idx * stride_r
        Y_ptr += row_idx * stride_r
        
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_cols
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        var = tl.sum(x * x, axis=0) / N_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd_ptr + row_idx, rstd)
        
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y_ptr + offsets, y.to(X_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_bwd_kernel(
        D_ptr, X_ptr, W_ptr, Rstd_ptr, DX_ptr, DW_ptr,
        stride_r, N_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        D_ptr += row_idx * stride_r
        X_ptr += row_idx * stride_r
        DX_ptr += row_idx * stride_r
        
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_cols
        
        dy = tl.load(D_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd_ptr + row_idx)
        
        dy_w = dy * w
        sum_dy_w_x = tl.sum(dy_w * x, axis=0)
        
        dx = rstd * dy_w - (rstd * rstd * rstd / N_cols) * x * sum_dy_w_x
        tl.store(DX_ptr + offsets, dx.to(DX_ptr.dtype.element_ty), mask=mask)
        
        dw = dy * x * rstd
        tl.store(DW_ptr + row_idx * BLOCK_SIZE + offsets, dw.to(DW_ptr.dtype.element_ty), mask=mask)

class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.eps = eps
        M, N = x.shape[0] * x.shape[1], x.shape[2]
        x_flat = x.view(M, N)
        y = torch.empty_like(x_flat)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        block_size = triton.next_power_of_2(N)
        _rmsnorm_fwd_kernel[(M,)](
            x_flat, y, weight, rstd,
            N, N, eps,
            BLOCK_SIZE=block_size
        )
        ctx.save_for_backward(x_flat, weight, rstd)
        return y.view_as(x)

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        M, N = x.shape
        dy_flat = dy.reshape(M, N)
        dx = torch.empty_like(x)
        block_size = triton.next_power_of_2(N)
        dw_buffer = torch.empty((M, block_size), dtype=torch.float32, device=x.device)
        
        _rmsnorm_bwd_kernel[(M,)](
            dy_flat, x, weight, rstd, dx, dw_buffer,
            N, N,
            BLOCK_SIZE=block_size
        )
        dw = dw_buffer[:, :N].sum(dim=0).to(weight.dtype)
        return dx.view_as(dy), dw, None

def triton_rmsnorm(x, weight, eps=1e-5):
    if _TRITON_AVAILABLE and x.is_cuda:
        return TritonRMSNormFunction.apply(x, weight, eps)
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# =====================================================================
# 2. FUSED LAYERNORM KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _layernorm_fwd_kernel(
        X_ptr, Y_ptr, W_ptr, B_ptr, Mean_ptr, Rstd_ptr,
        stride_r, N_cols, eps,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        X_ptr += row_idx * stride_r
        Y_ptr += row_idx * stride_r
        
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_cols
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        mean = tl.sum(x, axis=0) / N_cols
        tl.store(Mean_ptr + row_idx, mean)
        
        x_mu = x - mean
        var = tl.sum(x_mu * x_mu, axis=0) / N_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd_ptr + row_idx, rstd)
        
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x_mu * rstd * w + b
        tl.store(Y_ptr + offsets, y.to(Y_ptr.dtype.element_ty), mask=mask)

class TritonLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        M, N = x.shape[0] * x.shape[1], x.shape[2]
        x_flat = x.view(M, N)
        y = torch.empty_like(x_flat)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        block_size = triton.next_power_of_2(N)
        _layernorm_fwd_kernel[(M,)](
            x_flat, y, weight, bias, mean, rstd,
            N, N, eps,
            BLOCK_SIZE=block_size
        )
        ctx.save_for_backward(x_flat, weight, bias, mean, rstd)
        ctx.eps = eps
        return y.view_as(x)

    @staticmethod
    def backward(ctx, dy):
        # Fallback backprop for LayerNorm
        x, weight, bias, mean, rstd = ctx.saved_tensors
        M, N = x.shape
        dy_flat = dy.reshape(M, N)
        
        x_mu = x - mean.unsqueeze(-1)
        x_norm = x_mu * rstd.unsqueeze(-1)
        
        dw = (dy_flat * x_norm).sum(dim=0)
        db = dy_flat.sum(dim=0)
        
        # dx calculations
        dx_norm = dy_flat * weight
        dx = rstd.unsqueeze(-1) * (dx_norm - dx_norm.mean(dim=-1, keepdim=True) - x_norm * (dx_norm * x_norm).mean(dim=-1, keepdim=True))
        return dx.view_as(dy), dw, db, None

def triton_layernorm(x, weight, bias, eps=1e-5):
    if _TRITON_AVAILABLE and x.is_cuda:
        return TritonLayerNormFunction.apply(x, weight, bias, eps)
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

# =====================================================================
# 3. FUSED GLU ACTIVATION KERNELS
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _glu_fwd_kernel(
        Gate_ptr, Up_ptr, Out_ptr,
        N, type_flag,  # 0 for SiLU, 1 for GELU
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        gate = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        if type_flag == 0:
            # SwiGLU: SiLU(gate) * up
            act = gate * tl.sigmoid(gate)
        else:
            # GeGLU: GELU(gate) * up
            # Fast approximations for gelu
            act = gate * 0.5 * (1.0 + tl.math.erf(gate * 0.70710678))
            
        out = act * up
        tl.store(Out_ptr + offsets, out.to(Out_ptr.dtype.element_ty), mask=mask)

def triton_swiglu(gate, up):
    if _TRITON_AVAILABLE and gate.is_cuda:
        N = gate.numel()
        out = torch.empty_like(gate)
        block_size = 1024
        grid = (math.ceil(N / block_size),)
        _glu_fwd_kernel[grid](gate, up, out, N, 0, BLOCK_SIZE=block_size)
        return out
    return F.silu(gate) * up

def triton_geglu(gate, up):
    if _TRITON_AVAILABLE and gate.is_cuda:
        N = gate.numel()
        out = torch.empty_like(gate)
        block_size = 1024
        grid = (math.ceil(N / block_size),)
        _glu_fwd_kernel[grid](gate, up, out, N, 1, BLOCK_SIZE=block_size)
        return out
    return F.gelu(gate) * up

# =====================================================================
# 4. FUSED CROSS ENTROPY KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _cross_entropy_fwd_kernel(
        Logits_ptr, Targets_ptr, Loss_ptr,
        N_rows, N_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        if row_idx >= N_rows:
            return
            
        logits_row_ptr = Logits_ptr + row_idx * N_cols
        target_idx = tl.load(Targets_ptr + row_idx)
        
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_cols
        logits = tl.load(logits_row_ptr + offsets, mask=mask, other=-float('inf')).to(tl.float32)
        
        max_logit = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - max_logit)
        sum_exp = tl.sum(exp_logits, axis=0)
        log_sum_exp = tl.math.log(sum_exp) + max_logit
        
        target_logit = tl.load(logits_row_ptr + target_idx)
        loss = log_sum_exp - target_logit
        tl.store(Loss_ptr + row_idx, loss)

class TritonCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        N_rows, N_cols = logits.shape[0], logits.shape[1]
        loss = torch.empty((N_rows,), dtype=torch.float32, device=logits.device)
        
        block_size = triton.next_power_of_2(N_cols)
        _cross_entropy_fwd_kernel[(N_rows,)](
            logits, targets, loss,
            N_rows, N_cols,
            BLOCK_SIZE=block_size
        )
        ctx.save_for_backward(logits, targets)
        return loss.mean()

    @staticmethod
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        N_rows, N_cols = logits.shape[0], logits.shape[1]
        probs = torch.softmax(logits.float(), dim=-1)
        
        indices = torch.arange(N_rows, device=logits.device)
        probs[indices, targets] -= 1.0
        
        d_logits = (probs * d_loss) / N_rows
        return d_logits.to(logits.dtype), None

def triton_cross_entropy(logits, targets):
    if _TRITON_AVAILABLE and logits.is_cuda:
        original_shape = logits.shape
        logits_2d = logits.view(-1, original_shape[-1])
        targets_1d = targets.view(-1)
        return TritonCrossEntropyFunction.apply(logits_2d, targets_1d)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
