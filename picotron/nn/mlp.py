"""
SwiGLU, GeGLU, and Mixture of Experts (MoE) Feed-Forward Network (MLP) modules for Picotron.
Supports gated activation choices (silu/gelu), customizable linear bias settings, and MoE load-balancing loss.
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

    def forward(self, x: torch.Tensor):
        """Forward pass applying gating function (SwiGLU or GeGLU)."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        if self.activation_type == "gelu":
            act = F.gelu(gate) * up
        else:
            from picotron.kernels.triton_kernels import triton_swiglu
            act = triton_swiglu(gate, up)
            
        # Returns (output, auxiliary_loss=0.0) for unified block structure compatibility
        return self.down_proj(act), torch.tensor(0.0, device=x.device, dtype=x.dtype)


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module routing tokens across multiple GLU experts.
    Includes auxiliary load-balancing loss calculations to avoid expert collapse.
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

    def forward(self, x: torch.Tensor):
        """Route tokens through top-k experts and weight their outputs."""
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        
        # Compute router logits
        logits = self.gate(x_flat)
        
        # Select top-k experts
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        
        # Compute load balancing loss:
        # 1. Importance (average probability assigned to each expert)
        importance = scores.mean(0)
        # 2. Selection frequency (percentage of tokens routed to each expert)
        one_hot = F.one_hot(topk_indices, num_classes=self.num_experts).float() # [tokens, top_k, experts]
        routed = one_hot.sum(dim=1) # Sum across top_k selections -> [tokens, experts]
        frequency = routed.mean(0) # Average over batch -> [experts]
        
        # Auxiliary loss = N * sum(importance * frequency)
        aux_loss = self.num_experts * torch.sum(importance * frequency)
        
        # Normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        out = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx)
            token_indices, weight_ranks = torch.where(mask)
            
            if token_indices.numel() > 0:
                expert_inputs = x_flat[token_indices]
                expert_outputs, _ = self.experts[expert_idx](expert_inputs)
                
                weights = topk_weights[token_indices, weight_ranks].unsqueeze(-1)
                out.index_add_(0, token_indices, expert_outputs * weights)
                
        return out.view(*orig_shape), aux_loss
