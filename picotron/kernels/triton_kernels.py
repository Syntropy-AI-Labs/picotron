"""
High-performance custom Triton kernels for Picotron.
Implements fused RMSNorm, SwiGLU, RoPE, Cross Entropy, and AdamW.
Includes pure PyTorch fallback implementations for maximum compatibility.
"""

import math
import torch
import torch.nn.functional as F

# Check Triton availability
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
        stride_r,  # Row stride
        N_cols,    # Hidden size
        eps,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        X_ptr += row_idx * stride_r
        Y_ptr += row_idx * stride_r
        
        # Load inputs
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_cols
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute variance
        var = tl.sum(x * x, axis=0) / N_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        
        # Save reciprocal standard deviation for backward pass
        tl.store(Rstd_ptr + row_idx, rstd)
        
        # Load weights
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize and scale
        y = x * rstd * w
        tl.store(Y_ptr + offsets, y.to(X_ptr.dtype_element), mask=mask)

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
        
        # Load tensors
        dy = tl.load(D_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd_ptr + row_idx)
        
        # Mathematical gradient of RMSNorm
        # dx = rstd * w * dy - x * (rstd^3) / N * sum(dy * w * x)
        dy_w = dy * w
        sum_dy_w_x = tl.sum(dy_w * x, axis=0)
        
        dx = rstd * dy_w - (rstd * rstd * rstd / N_cols) * x * sum_dy_w_x
        tl.store(DX_ptr + offsets, dx.to(DX_ptr.dtype_element), mask=mask)
        
        # DW update is computed globally or via atomic additions
        dw = dy * x * rstd
        tl.store(DW_ptr + row_idx * BLOCK_SIZE + offsets, dw.to(DW_ptr.dtype_element), mask=mask)

class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.eps = eps
        M, N = x.shape[0] * x.shape[1], x.shape[2]
        x_flat = x.view(M, N)
        
        y = torch.empty_like(x_flat)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        
        # Auto-tune block size
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
        
        # Buffer to aggregate weight gradients across parallel rows
        dw_buffer = torch.empty((M, block_size), dtype=torch.float32, device=x.device)
        
        _rmsnorm_bwd_kernel[(M,)](
            dy_flat, x, weight, rstd, dx, dw_buffer,
            N, N,
            BLOCK_SIZE=block_size
        )
        
        dw = dw_buffer[:, :N].sum(dim=0).to(weight.dtype)
        return dx.view_as(dy), dw, None

def triton_rmsnorm(x, weight, eps=1e-5):
    """Call optimized Triton Fused RMSNorm, falling back to PyTorch."""
    if _TRITON_AVAILABLE and x.is_cuda:
        return TritonRMSNormFunction.apply(x, weight, eps)
    # Fast PyTorch fallback
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# =====================================================================
# 2. FUSED SWIGLU KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _swiglu_fwd_kernel(
        Gate_ptr, Up_ptr, Out_ptr,
        N,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        gate = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Fused SiLU(gate) * up
        silu_gate = gate * tl.sigmoid(gate)
        out = silu_gate * up
        
        tl.store(Out_ptr + offsets, out.to(Out_ptr.dtype_element), mask=mask)

def triton_swiglu(gate, up):
    """Fused SwiGLU operation (silu(gate) * up), falling back to PyTorch."""
    if _TRITON_AVAILABLE and gate.is_cuda:
        N = gate.numel()
        out = torch.empty_like(gate)
        block_size = 1024
        grid = (math.ceil(N / block_size),)
        
        _swiglu_fwd_kernel[grid](
            gate, up, out,
            N, BLOCK_SIZE=block_size
        )
        return out
    return F.silu(gate) * up if hasattr(torch.nn.functional, "silu") else (gate * torch.sigmoid(gate)) * up

# =====================================================================
# 3. FUSED ROPE KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _rope_fwd_kernel(
        Q_ptr, Cos_ptr, Sin_ptr, Out_ptr,
        seq_len, num_heads, head_dim,
        stride_b, stride_s, stride_h, stride_d,
        BLOCK_SIZE: tl.constexpr
    ):
        # Index layout: Batch (x), Seq (y), Head (z)
        batch_idx = tl.program_id(0)
        seq_idx = tl.program_id(1)
        head_idx = tl.program_id(2)
        
        # Load cosine and sine indices
        cos_offset = seq_idx * head_dim
        offsets = tl.arange(0, BLOCK_SIZE)
        half_dim = head_dim // 2
        
        cos_mask = offsets < half_dim
        cos = tl.load(Cos_ptr + cos_offset + offsets, mask=cos_mask, other=0.0).to(tl.float32)
        sin = tl.load(Sin_ptr + cos_offset + offsets, mask=cos_mask, other=0.0).to(tl.float32)
        
        # Load Q values (first half and second half)
        q_offset = batch_idx * stride_b + seq_idx * stride_s + head_idx * stride_h
        
        q1 = tl.load(Q_ptr + q_offset + offsets, mask=cos_mask, other=0.0).to(tl.float32)
        q2 = tl.load(Q_ptr + q_offset + half_dim + offsets, mask=cos_mask, other=0.0).to(tl.float32)
        
        # Fused rotation:
        # out1 = q1 * cos - q2 * sin
        # out2 = q1 * sin + q2 * cos
        out1 = q1 * cos - q2 * sin
        out2 = q1 * sin + q2 * cos
        
        # Store outputs
        tl.store(Out_ptr + q_offset + offsets, out1.to(Out_ptr.dtype_element), mask=cos_mask)
        tl.store(Out_ptr + q_offset + half_dim + offsets, out2.to(Out_ptr.dtype_element), mask=cos_mask)

def triton_rope(q, cos, sin):
    """Fused Rotary Positional Embedding application using Triton, falling back to PyTorch."""
    if _TRITON_AVAILABLE and q.is_cuda:
        bsz, seq_len, num_heads, head_dim = q.shape
        out = torch.empty_like(q)
        
        stride_b, stride_s, stride_h, stride_d = q.stride()
        block_size = triton.next_power_of_2(head_dim // 2)
        grid = (bsz, seq_len, num_heads)
        
        _rope_fwd_kernel[grid](
            q, cos, sin, out,
            seq_len, num_heads, head_dim,
            stride_b, stride_s, stride_h, stride_d,
            BLOCK_SIZE=block_size
        )
        return out
    
    # PyTorch fallback (handles Broadcasting alignment)
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    cos_sliced = cos[:, :half_dim].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, half_dim]
    sin_sliced = sin[:, :half_dim].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, half_dim]
    return torch.cat([q1 * cos_sliced - q2 * sin_sliced, q1 * sin_sliced + q2 * cos_sliced], dim=-1)

# =====================================================================
# 4. FUSED ADAMW KERNEL
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
        
        # Load parameters and states
        p = tl.load(Param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(Grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(M_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Weight decay update
        p = p * (1.0 - lr * wd)
        
        # Update moment vectors
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        
        # Save back moments
        tl.store(M_ptr + offsets, m, mask=mask)
        tl.store(V_ptr + offsets, v, mask=mask)
        
        # Bias corrections
        bias_corr1 = 1.0 - tl.math.pow(beta1, step)
        bias_corr2 = 1.0 - tl.math.pow(beta2, step)
        
        step_sz = lr * tl.math.sqrt(bias_corr2) / bias_corr1
        
        # Update weights
        p_new = p - step_sz * m / (tl.math.sqrt(v) + eps)
        tl.store(Param_ptr + offsets, p_new.to(Param_ptr.dtype_element), mask=mask)

def triton_adamw(params, grads, exp_avgs, exp_avg_sqs, lr, beta1, beta2, eps, wd, step):
    """Triton-accelerated fused AdamW optimizer update step, falling back to PyTorch."""
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
        
    # PyTorch Fallback
    params.mul_(1 - lr * wd)
    exp_avgs.mul_(beta1).add_(grads, alpha=1 - beta1)
    exp_avg_sqs.mul_(beta2).addcmul_(grads, grads, value=1 - beta2)
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    denom = exp_avg_sqs.sqrt().add_(eps)
    params.addcdiv_(exp_avgs, denom, value=-step_size)

# =====================================================================
# 5. FUSED CROSS ENTROPY KERNEL
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
    """Fused Cross Entropy Loss, falling back to PyTorch."""
    if _TRITON_AVAILABLE and logits.is_cuda:
        original_shape = logits.shape
        logits_2d = logits.view(-1, original_shape[-1])
        targets_1d = targets.view(-1)
        return TritonCrossEntropyFunction.apply(logits_2d, targets_1d)
        
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

