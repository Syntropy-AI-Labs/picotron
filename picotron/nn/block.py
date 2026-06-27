"""
Transformer Block for Picotron.
Supports pre-norm, GQA/MLA attention, SwiGLU or MoE FFN, and parallel execution paths.
"""

import torch
import torch.nn as nn
from picotron.config import ModelConfig
from picotron.nn.norm import RMSNorm
from picotron.nn.attention import Attention
from picotron.nn.mlp import MLP, MoE

class TransformerBlock(nn.Module):
    """
    Highly configurable Transformer Block.
    """
    def __init__(self, config: ModelConfig):
        """Initialize block normalization, attention, and FFN modules from config."""
        super().__init__()
        self.config = config
        self.parallel_attn_ffn = config.parallel_attn_ffn
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 1. Attention Module
        num_kv = config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv,
            qk_norm=config.qk_norm,
            use_mla=config.use_mla,
            mla_kv_lora_rank=config.mla_kv_lora_rank,
            mla_qk_lora_rank=config.mla_qk_lora_rank,
            mla_qk_rope_lora_rank=config.mla_qk_rope_lora_rank,
            sliding_window=config.sliding_window,
            logit_soft_cap=config.logit_soft_cap,
            rms_norm_eps=config.rms_norm_eps,
        )
        
        # Calculate intermediate dimension
        if config.intermediate_size is None:
            intermediate_size = int(2 * (4 * config.hidden_size) / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        else:
            intermediate_size = config.intermediate_size

        # 2. FFN Module: standard SwiGLU or Mixture of Experts (MoE)
        if config.use_moe:
            self.mlp = MoE(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                num_experts=config.moe_num_experts,
                top_k=config.moe_top_k
            )
        else:
            self.mlp = MLP(hidden_size=config.hidden_size, intermediate_size=intermediate_size)
            
        # Post-attn norm is only required/used if parallel attention and FFN is disabled
        if not self.parallel_attn_ffn:
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        """Forward pass through block (sequential pre-norm or parallel attention/FFN)."""
        if self.parallel_attn_ffn:
            # Parallel attention and FFN: run simultaneously, sum outputs
            residual = hidden_states
            normed = self.input_layernorm(hidden_states)
            
            attn_out = self.self_attn(normed, cos, sin, rotary_emb)
            ffn_out = self.mlp(normed)
            
            hidden_states = residual + attn_out + ffn_out
        else:
            # Standard sequential pre-norm path
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states, cos, sin, rotary_emb)
            hidden_states = residual + hidden_states
            
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
        return hidden_states
