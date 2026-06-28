"""
Multi-Head, Grouped-Query (GQA), and Multi-Head Latent Attention (MLA) module for Picotron.
Supports QK-Norm, sliding window masking, logit soft-capping, and FlashAttn2 detection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from picotron.nn.norm import get_norm

# Try importing flash_attn at module level safely
_FLASH_ATTN_AVAILABLE = False
try:
    import flash_attn
    from flash_attn import flash_attn_func
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            _FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

class Attention(nn.Module):
    """
    Highly configurable Attention module supporting GQA, MLA, QK-Norm, logit soft capping, and sliding window attention.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int = None,
        dropout: float = 0.0,
        qk_norm: bool = False,
        use_mla: bool = False,
        mla_kv_lora_rank: int = 512,
        mla_qk_lora_rank: int = 128,
        mla_qk_rope_lora_rank: int = 64,
        sliding_window: int = None,
        logit_soft_cap: float = None,
        rms_norm_eps: float = 1e-5,
        bias: bool = False,
        norm_type: str = "rms",
        layer_idx: int = 0,
        alternate_sliding_window: bool = False,
    ):
        """Initialize attention components."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        self.qk_norm = qk_norm
        self.use_mla = use_mla
        self.sliding_window = sliding_window
        self.logit_soft_cap = logit_soft_cap
        self.dropout = dropout
        self.layer_idx = layer_idx
        self.alternate_sliding_window = alternate_sliding_window

        # Decoupled RoPE dimension
        self.rope_dim = self.head_dim

        if self.use_mla:
            # MLA uses compressed latent KV representation
            self.kv_lora_rank = mla_kv_lora_rank
            self.qk_lora_rank = mla_qk_lora_rank
            self.qk_rope_lora_rank = mla_qk_rope_lora_rank
            
            # Compressed projections
            self.kv_down_proj = nn.Linear(hidden_size, self.kv_lora_rank, bias=bias)
            self.kv_up_proj = nn.Linear(self.kv_lora_rank, num_attention_heads * (self.head_dim + self.qk_rope_lora_rank), bias=bias)
            
            self.q_down_proj = nn.Linear(hidden_size, self.qk_lora_rank, bias=bias)
            self.q_up_proj = nn.Linear(self.qk_lora_rank, num_attention_heads * (self.head_dim + self.qk_rope_lora_rank), bias=bias)
            
            # Decoupled key for RoPE
            self.k_rope_proj = nn.Linear(hidden_size, mla_qk_rope_lora_rank, bias=bias)
            
            # Decoupled RoPE dimensions
            self.rope_dim = self.qk_rope_lora_rank
            self.total_q_dim = self.head_dim + self.qk_rope_lora_rank
        else:
            self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=bias)
            self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
            self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)

        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=bias)

        # Optional QK-Norm LayerNorms
        if self.qk_norm:
            self.q_ln = get_norm(self.head_dim, norm_type=norm_type, eps=rms_norm_eps)
            self.k_ln = get_norm(self.head_dim, norm_type=norm_type, eps=rms_norm_eps)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key or value states along the head dimension for Grouped-Query Attention."""
        if n_rep == 1:
            return x
        bsz, num_kv_heads, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
            .reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        """Forward pass executing Multi-Head, GQA or MLA attention."""
        bsz, seq_len, _ = hidden_states.shape
        
        if self.use_mla:
            # 1. Project Query
            q_latent = self.q_down_proj(hidden_states)
            q_all = self.q_up_proj(q_latent).view(bsz, seq_len, self.num_attention_heads, self.total_q_dim).transpose(1, 2)
            
            # Split Q into main attention Q (no RoPE/identity) and RoPE Q
            q_main = q_all[..., :self.head_dim]
            q_rope = q_all[..., self.head_dim:]
            
            # 2. Project KV
            kv_latent = self.kv_down_proj(hidden_states)
            kv_all = self.kv_up_proj(kv_latent).view(bsz, seq_len, self.num_attention_heads, self.total_q_dim).transpose(1, 2)
            
            k_main = kv_all[..., :self.head_dim]
            v = kv_all[..., self.head_dim:]  # GQA style repeated or directly head dimension
            
            # Decoupled key for RoPE
            k_rope = self.k_rope_proj(hidden_states).view(bsz, seq_len, 1, self.qk_rope_lora_rank).transpose(1, 2)
            k_rope = k_rope.expand(-1, self.num_attention_heads, -1, -1)  # Expand key rope across all heads
            
            # Apply rotary positional embeddings only to the decoupled rope dimensions
            q_rope, k_rope = rotary_emb.apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
            
            # Combine main and rope paths
            q = torch.cat([q_main, q_rope], dim=-1)
            k = torch.cat([k_main, k_rope], dim=-1)
        else:
            q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            # Apply QK-Norm if enabled
            if self.qk_norm:
                q = self.q_ln(q)
                k = self.k_ln(k)

            # Apply standard RoPE
            q, k = rotary_emb.apply_rotary_pos_emb(q, k, cos, sin)
            
            # Repeat KV if GQA is active
            n_rep = self.num_attention_heads // self.num_key_value_heads
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)

        # Pad head dim if required (must be divisible by 8)
        curr_head_dim = q.size(-1)
        pad_len = 0
        if curr_head_dim % 8 != 0:
            pad_len = 8 - (curr_head_dim % 8)
            q = F.pad(q, (0, pad_len))
            k = F.pad(k, (0, pad_len))
            v = F.pad(v, (0, pad_len))

        # Check flash attention availability
        # For alternating sliding window, we check if it is active for this layer
        sliding_window = self.sliding_window
        if self.alternate_sliding_window and (self.layer_idx % 2 != 0):
            sliding_window = None

        flash_available = _FLASH_ATTN_AVAILABLE and (q.device.type == "cuda") and (sliding_window is None) and (self.logit_soft_cap is None)
        
        if flash_available:
            q_flash = q.transpose(1, 2)
            k_flash = k.transpose(1, 2)
            v_flash = v.transpose(1, 2)
            attn_output = flash_attn_func(
                q_flash, k_flash, v_flash,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
            attn_output = attn_output.reshape(bsz, seq_len, -1)
        else:
            # Fallback to manual SDPA with soft cap or sliding window
            # Compute attention score matrix
            # Q: [B, H, S, D], K: [B, H, S, D] -> [B, H, S, S]
            q = q / math.sqrt(q.size(-1))
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            
            # Apply logit soft cap
            if self.logit_soft_cap is not None:
                attn_weights = self.logit_soft_cap * torch.tanh(attn_weights / self.logit_soft_cap)
                
            # Create causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).view(1, 1, seq_len, seq_len)
            
            # Apply sliding window masking
            if sliding_window is not None:
                window_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=-sliding_window + 1)
                causal_mask = causal_mask * window_mask.view(1, 1, seq_len, seq_len)
                
            mask_value = torch.finfo(attn_weights.dtype).min
            attn_weights = torch.where(causal_mask == 1, attn_weights, torch.tensor(mask_value, device=q.device, dtype=attn_weights.dtype))
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.training and self.dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
                
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
            
        # Crop the output back to original head_dim if padded
        if pad_len > 0:
            attn_output = attn_output.view(bsz, seq_len, self.num_attention_heads, curr_head_dim + pad_len)
            attn_output = attn_output[..., :curr_head_dim].reshape(bsz, seq_len, -1)
            
        return self.o_proj(attn_output)
