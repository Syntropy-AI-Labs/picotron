"""
Triton attention operations: RoPE, Softmax, and Dropout kernels.
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
# 1. FUSED ROPE KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _rope_fwd_kernel(
        Q_ptr, Cos_ptr, Sin_ptr, Out_ptr,
        seq_len, num_heads, head_dim,
        stride_b, stride_s, stride_h, stride_d,
        BLOCK_SIZE: tl.constexpr
    ):
        batch_idx = tl.program_id(0)
        seq_idx = tl.program_id(1)
        head_idx = tl.program_id(2)
        
        cos_offset = seq_idx * head_dim
        offsets = tl.arange(0, BLOCK_SIZE)
        half_dim = head_dim // 2
        
        cos_mask = offsets < half_dim
        cos = tl.load(Cos_ptr + cos_offset + offsets, mask=cos_mask, other=0.0).to(tl.float32)
        sin = tl.load(Sin_ptr + cos_offset + offsets, mask=cos_mask, other=0.0).to(tl.float32)
        
        q_offset = batch_idx * stride_b + seq_idx * stride_s + head_idx * stride_h
        
        q1 = tl.load(Q_ptr + q_offset + offsets, mask=cos_mask, other=0.0).to(tl.float32)
        q2 = tl.load(Q_ptr + q_offset + half_dim + offsets, mask=cos_mask, other=0.0).to(tl.float32)
        
        out1 = q1 * cos - q2 * sin
        out2 = q1 * sin + q2 * cos
        
        tl.store(Out_ptr + q_offset + offsets, out1.to(Out_ptr.dtype.element_ty), mask=cos_mask)
        tl.store(Out_ptr + q_offset + half_dim + offsets, out2.to(Out_ptr.dtype.element_ty), mask=cos_mask)

def triton_rope(q, cos, sin):
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
    
    half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    cos_sliced = cos[:, :half_dim].unsqueeze(0).unsqueeze(2)
    sin_sliced = sin[:, :half_dim].unsqueeze(0).unsqueeze(2)
    return torch.cat([q1 * cos_sliced - q2 * sin_sliced, q1 * sin_sliced + q2 * cos_sliced], dim=-1)

# =====================================================================
# 2. FUSED SOFTMAX KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _softmax_fwd_kernel(
        X_ptr, Y_ptr,
        N_rows, N_cols,
        BLOCK_SIZE: tl.constexpr
    ):
        row_idx = tl.program_id(0)
        if row_idx >= N_rows:
            return
            
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_cols
        x = tl.load(X_ptr + row_idx * N_cols + offsets, mask=mask, other=-float('inf')).to(tl.float32)
        
        max_val = tl.max(x, axis=0)
        exp_x = tl.exp(x - max_val)
        sum_exp = tl.sum(exp_x, axis=0)
        
        y = exp_x / sum_exp
        tl.store(Y_ptr + row_idx * N_cols + offsets, y.to(Y_ptr.dtype.element_ty), mask=mask)

def triton_softmax(x):
    if _TRITON_AVAILABLE and x.is_cuda:
        original_shape = x.shape
        x_2d = x.view(-1, original_shape[-1])
        N_rows, N_cols = x_2d.shape
        out = torch.empty_like(x_2d)
        
        block_size = triton.next_power_of_2(N_cols)
        _softmax_fwd_kernel[(N_rows,)](x_2d, out, N_rows, N_cols, BLOCK_SIZE=block_size)
        return out.view(original_shape)
    return F.softmax(x, dim=-1)

# =====================================================================
# 3. FUSED DROPOUT KERNEL
# =====================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _dropout_kernel(
        X_ptr, Out_ptr, seed, p,
        N,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Simple pseudo-random generation using offset seeds
        rand = tl.rand(seed, offsets)
        keep = rand >= p
        
        out = tl.where(keep, x / (1.0 - p), 0.0)
        tl.store(Out_ptr + offsets, out.to(Out_ptr.dtype.element_ty), mask=mask)

def triton_dropout(x, p=0.0, training=True):
    if not training or p == 0.0:
        return x
    if _TRITON_AVAILABLE and x.is_cuda:
        N = x.numel()
        out = torch.empty_like(x)
        block_size = 1024
        grid = (math.ceil(N / block_size),)
        
        seed = torch.randint(0, 65535, (1,)).item()
        _dropout_kernel[grid](x, out, seed, p, N, BLOCK_SIZE=block_size)
        return out
    return F.dropout(x, p=p, training=training)
