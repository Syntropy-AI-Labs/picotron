"""
Rotary Position Embedding (RoPE) implementation with support for NoPE (no rotation)
and YaRN-style scaling (context length extension).
"""

import math
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module supporting NoPE and scaling.
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        position_embedding_type: str = "rope",
        scaling_factor: float = 1.0
    ):
        """Initialize RoPE caches with scaling options."""
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.position_embedding_type = position_embedding_type
        self.scaling_factor = scaling_factor

        if self.position_embedding_type == "nope":
            return

        # Precompute cos and sin frequencies
        # For YaRN/Linear scaling, we scale the frequencies by the scaling factor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        if self.position_embedding_type == "yarn" or self.scaling_factor != 1.0:
            inv_freq = inv_freq / self.scaling_factor

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        
        # [max_seq_len, dim] -> cos/sin of shape [max_seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input tensor."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the precomputed cos and sin slices corresponding to seq_len."""
        if self.position_embedding_type == "nope":
            return None, None
        cos = self.cos_cached[:seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len, :].to(dtype=x.dtype, device=x.device)
        return cos, sin

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        if self.position_embedding_type == "nope" or cos is None or sin is None:
            return q, k
            
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
