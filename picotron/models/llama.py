"""
LLaMA Decoder Model for Picotron.
Assembles embedding, transformer layers, rotary/NoPE embeddings, and head projection.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from picotron.config import ModelConfig
from picotron.nn.norm import RMSNorm
from picotron.nn.rope import RotaryEmbedding
from picotron.nn.block import TransformerBlock

class LLaMAModel(nn.Module):
    """
    LLaMA style autoregressive decoder transformer with support for modern attention models.
    """
    def __init__(self, config: ModelConfig):
        """Initialize the network components from configuration."""
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Word embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Rotary Positional Embeddings
        # If MLA is enabled, the decoupled key/query rope dimensions may differ, handled in Attention module
        rope_dim = config.hidden_size // config.num_attention_heads
        if config.use_mla:
            rope_dim = config.mla_qk_rope_lora_rank

        self.rotary_emb = RotaryEmbedding(
            dim=rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            position_embedding_type=config.position_embedding_type,
            scaling_factor=config.rope_scaling_factor
        )

        # Decoder blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config=config)
            for _ in range(config.num_hidden_layers)
        ])

        # Final pre-head RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language Model Head (tied/untied option)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights standard to LLaMA training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through LLaMA decoder."""
        _, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Precompute rotary cos and sin slices for current seq_len
        cos, sin = self.rotary_emb(hidden_states, seq_len)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                rotary_emb=self.rotary_emb
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
