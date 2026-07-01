"""
Transformer Block for Picotron.
Supports pre-norm, GQA/MLA attention, Gated DeltaNet (Qwen 3.5), SwiGLU or MoE FFN, and parallel execution paths.
"""

import torch
import torch.nn as nn
from picotron.config import ModelConfig
from picotron.nn.norm import get_norm
from picotron.nn.attention import Attention
from picotron.nn.mlp import MLP, MoE
from picotron.nn.deltanet import GatedDeltaNet

class TransformerBlock(nn.Module):
    """
    Highly configurable Transformer Block.
    """
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        """Initialize block normalization, attention, and FFN modules from config."""
        super().__init__()
        self.config = config
        self.parallel_attn_ffn = config.parallel_attn_ffn
        self.layer_idx = layer_idx
        
        self.input_layernorm = get_norm(config.hidden_size, norm_type=config.norm_type, eps=config.rms_norm_eps)
        
        # 1. Hybrid Gated DeltaNet or standard Self-Attention selection
        self.is_deltanet = False
        if config.use_deltanet:
            # Alternate: e.g. 3 DeltaNet layers followed by 1 standard Attention layer
            if (layer_idx % (config.deltanet_ratio + 1)) < config.deltanet_ratio:
                self.is_deltanet = True

        if self.is_deltanet:
            self.self_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                bias=config.bias
            )
        elif config.mixer_type == "mla":
            from picotron.nn.mixers import MLA
            head_dim = config.hidden_size // config.num_attention_heads
            self.self_attn = MLA(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=head_dim
            )
        elif config.mixer_type == "mamba":
            from picotron.nn.mixers import SelectiveSSM
            self.self_attn = SelectiveSSM(
                d_model=config.hidden_size,
                d_inner=config.hidden_size
            )
        elif config.mixer_type == "rwkv":
            from picotron.nn.mixers import RWKVRecurrence
            self.self_attn = RWKVRecurrence(
                d_model=config.hidden_size,
                d_inner=config.hidden_size
            )
        else:
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
                bias=config.bias,
                norm_type=config.norm_type,
                layer_idx=layer_idx,
                alternate_sliding_window=config.alternate_sliding_window,
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
                top_k=config.moe_top_k,
                activation_type=config.activation_type,
                bias=config.bias
            )
        else:
            self.mlp = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                activation_type=config.activation_type,
                bias=config.bias
            )
            
        # Post-attn norm is only required/used if parallel attention and FFN is disabled
        if not self.parallel_attn_ffn:
            self.post_attention_layernorm = get_norm(config.hidden_size, norm_type=config.norm_type, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary_emb: nn.Module,
    ):
        """Forward pass returning hidden states and auxiliary routing losses."""
        # Select correct mixer forwarding signature
        def forward_mixer(layer, input_tensor):
            if self.config.mixer_type == "mamba" or self.config.mixer_type == "rwkv":
                return layer(input_tensor)
            elif self.config.mixer_type == "mla":
                return layer(input_tensor, cos, sin)
            elif self.is_deltanet:
                return layer(input_tensor)
            else:
                return layer(input_tensor, cos, sin, rotary_emb)

        if self.parallel_attn_ffn:
            # Parallel attention/DeltaNet and FFN: run simultaneously, sum outputs
            residual = hidden_states
            normed = self.input_layernorm(hidden_states)
            
            attn_out = forward_mixer(self.self_attn, normed)
            ffn_out, aux_loss = self.mlp(normed)
            hidden_states = residual + attn_out + ffn_out
        else:
            # Standard sequential pre-norm path
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            
            attn_out = forward_mixer(self.self_attn, hidden_states)
            hidden_states = residual + attn_out
            
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            ffn_out, aux_loss = self.mlp(hidden_states)
            hidden_states = residual + ffn_out
            
        return hidden_states, aux_loss
