"""
SwiGLU, GeGLU, and Mixture of Experts (MoE) Feed-Forward Network (MLP) modules for Picotron.
Supports gated activation choices (silu/gelu) and customizable linear bias settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Gated Linear Unit (GLU) Feed-Forward Network.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, activation_type: str = "silu", bias: bool = False):
        """Initialize GLU projections and activation mappings."""
        super().__init__()
        self.activation_type = activation_type
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying gating function (SwiGLU or GeGLU)."""
        if self.activation_type == "gelu":
            act = F.gelu(self.gate_proj(x))
        else:
            act = F.silu(self.gate_proj(x))
            
        return self.down_proj(act * self.up_proj(x))


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module routing tokens across multiple GLU experts.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 8, top_k: int = 2, activation_type: str = "silu", bias: bool = False):
        """Initialize experts and the gating router."""
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Instantiate GLU experts
        self.experts = nn.ModuleList([
            MLP(hidden_size, intermediate_size, activation_type=activation_type, bias=bias) 
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens through top-k experts and weight their outputs."""
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        
        # Compute router logits
        logits = self.gate(x_flat)
        
        # Select top-k experts
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        
        # Normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        out = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx)
            token_indices, weight_ranks = torch.where(mask)
            
            if token_indices.numel() > 0:
                expert_inputs = x_flat[token_indices]
                expert_outputs = self.experts[expert_idx](expert_inputs)
                
                weights = topk_weights[token_indices, weight_ranks].unsqueeze(-1)
                out.index_add_(0, token_indices, expert_outputs * weights)
                
        return out.view(*orig_shape)
