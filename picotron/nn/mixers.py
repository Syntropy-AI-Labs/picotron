"""
Advanced sequence mixers for Picotron:
MLA (Multi-Head Latent Attention), Selective State Space Models (Mamba-style), and RWKV-style time-decay recurrences.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# =====================================================================
# 1. MULTI-HEAD LATENT ATTENTION (MLA) - DeepSeek V2/V3 style
# =====================================================================
class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) implementation.
    Compresses KV cache using low-rank projection to bypass cache capacity limits.
    Decouples rotary positional embeddings to keep positional signals separate from latent representations.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 128,
        rope_dim: int = 64
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim
        
        # Q Projection compression
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
        
        # KV Projection compression
        self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (head_dim + rope_dim), bias=False)
        
        # decoupled RoPE projections
        self.q_rope_proj = nn.Linear(q_lora_rank, num_heads * rope_dim, bias=False)
        
        # Out projection
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        
        # 1. Compress & project Queries
        q_latent = self.q_a_proj(x) # [bsz, seq_len, q_lora_rank]
        q = self.q_b_proj(q_latent).view(bsz, seq_len, self.num_heads, self.head_dim)
        
        # Decoupled Query RoPE projection
        q_rope = self.q_rope_proj(q_latent).view(bsz, seq_len, self.num_heads, self.rope_dim)
        
        # 2. Compress & project KV
        kv_latent = self.kv_a_proj(x) # [bsz, seq_len, kv_lora_rank]
        kv = self.kv_b_proj(kv_latent).view(bsz, seq_len, self.num_heads, self.head_dim + self.rope_dim)
        
        # Split KV into base keys/values and keys for RoPE
        k = kv[..., :self.head_dim]
        k_rope = kv[..., self.head_dim:]
        v = k  # MLA shares representations for values
        
        # 3. Apply RoPE on decoupled parts if cos/sin are available
        if cos is not None and sin is not None:
            # Reshape cos/sin to match heads dimension
            c = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, rope_dim]
            s = sin[:seq_len].unsqueeze(0).unsqueeze(2)
            
            # Apply rotary transformation to rope channels
            half = self.rope_dim // 2
            q1, q2 = q_rope[..., :half], q_rope[..., half:]
            q_rope = torch.cat([-q2, q1], dim=-1) * s + q_rope * c
            
            k1, k2 = k_rope[..., :half], k_rope[..., half:]
            k_rope = torch.cat([-k2, k1], dim=-1) * s + k_rope * c
            
        # 4. Attention mechanism (incorporating decoupled RoPE)
        # We concatenate base key/query vectors with their rotary-embedded decoupled parts
        q_attn = torch.cat([q, q_rope], dim=-1).transpose(1, 2)  # [B, H, S, D + R]
        k_attn = torch.cat([k, k_rope], dim=-1).transpose(1, 2)  # [B, H, S, D + R]
        v_attn = v.transpose(1, 2)  # [B, H, S, D]
        
        # Scaled dot-product attention
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) / math.sqrt(q_attn.size(-1))
        attn_probs = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_probs, v_attn) # [B, H, S, D]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        
        return self.out_proj(out)

# =====================================================================
# 2. SELECTIVE STATE SPACE MODEL (Mamba-style)
# =====================================================================
class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (SSM) layer.
    Features input-dependent time-steps (delta), selective transitions (A, B), and outputs (C).
    Allows sequence mixers to execute with O(N) context length scaling.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_inner: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        
        # Dynamic selectivity projections
        self.x_proj = nn.Linear(d_inner, d_state + d_state + d_inner, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        
        # State transitions parameters
        self.A = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [bsz, seq_len, d_inner]
        bsz, seq_len, _ = x.shape
        
        # 1. Project to get selective parameters: delta, B, C
        proj = self.x_proj(x)
        B, C, dt_raw = torch.split(proj, [self.d_state, self.d_state, self.d_inner], dim=-1)
        
        # Compute time steps delta (discretizer step)
        dt = F.softplus(self.dt_proj(dt_raw)) # [bsz, seq_len, d_inner]
        
        # 2. Discretization using Euler approach:
        # dA = exp(dt * A)
        # dB = dt * B
        A = -torch.exp(self.A) # [d_inner, d_state]
        
        # Recurrent calculation loop
        h = torch.zeros((bsz, self.d_inner, self.d_state), device=x.device, dtype=x.dtype)
        y = torch.empty_like(x)
        
        for t in range(seq_len):
            xt = x[:, t, :] # [bsz, d_inner]
            dt_t = dt[:, t, :].unsqueeze(-1) # [bsz, d_inner, 1]
            bt_t = B[:, t, :].unsqueeze(1) # [bsz, 1, d_state]
            ct_t = C[:, t, :].unsqueeze(-1) # [bsz, d_state, 1]
            
            # Discretize state matrices
            dA = torch.exp(dt_t * A.unsqueeze(0)) # [bsz, d_inner, d_state]
            dB = dt_t * bt_t # [bsz, d_inner, d_state]
            
            # Recurrence transition
            h = dA * h + dB * xt.unsqueeze(-1)
            
            # Compute output
            yt = torch.matmul(h, ct_t).squeeze(-1) # [bsz, d_inner]
            y[:, t, :] = yt + xt * self.D.unsqueeze(0)
            
        return y

# =====================================================================
# 3. RWKV RECURRENCE MODULE (RWKV-style time decay)
# =====================================================================
class RWKVRecurrence(nn.Module):
    """
    RWKV-style linear recurrent attention block.
    Features channel-wise decay factors (W) and keys-values gating.
    """
    def __init__(self, d_model: int, d_inner: int = 128):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        
        # Projection layers
        self.time_mix_k = nn.Parameter(torch.rand(d_inner))
        self.time_mix_v = nn.Parameter(torch.rand(d_inner))
        self.time_mix_r = nn.Parameter(torch.rand(d_inner))
        
        # Decay rates and parameters
        self.time_decay = nn.Parameter(torch.randn(d_inner))
        self.time_first = nn.Parameter(torch.randn(d_inner))
        
        # Linear outputs
        self.key_proj = nn.Linear(d_model, d_inner, bias=False)
        self.value_proj = nn.Linear(d_model, d_inner, bias=False)
        self.receptance_proj = nn.Linear(d_model, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        
        k = self.key_proj(x)
        v = self.value_proj(x)
        r = self.receptance_proj(x)
        
        # Recurrent calculation using time decay factors
        # h_t = decay * h_{t-1} + key * value
        h = torch.zeros((bsz, self.d_inner), device=x.device, dtype=x.dtype)
        y = torch.empty_like(r)
        
        decay = torch.exp(-torch.exp(self.time_decay)).unsqueeze(0) # [1, d_inner]
        first = torch.exp(self.time_first).unsqueeze(0)
        
        for t in range(seq_len):
            kt = k[:, t, :]
            vt = v[:, t, :]
            rt = r[:, t, :]
            
            # RWKV recurrent state transition
            h_new = decay * h + kt * vt
            
            # Apply time-first factor
            output_t = torch.sigmoid(rt) * (first * h + kt * vt)
            y[:, t, :] = output_t
            h = h_new
            
        return self.out_proj(y)
