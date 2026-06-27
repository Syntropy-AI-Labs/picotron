"""
SwiGLU Feed-Forward Network (MLP) and Mixture of Experts (MoE) modules for Picotron.
Supports standard LLaMA-style FFN layers and Top-K routing across multiple experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        """Initialize SwiGLU projection layers."""
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying SwiGLU activation."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) module routing tokens across multiple SwiGLU experts.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 8, top_k: int = 2):
        """Initialize experts and the gating router."""
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Instantiate SwiGLU experts
        self.experts = nn.ModuleList([
            MLP(hidden_size, intermediate_size) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens through top-k experts and weight their outputs."""
        # x shape: [batch_size, seq_len, hidden_size]
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
        
        # Route each token to its selected experts
        for expert_idx in range(self.num_experts):
            # Check which tokens are routed to this expert
            mask = (topk_indices == expert_idx)
            token_indices, weight_ranks = torch.where(mask)
            
            if token_indices.numel() > 0:
                # Gather inputs for this expert
                expert_inputs = x_flat[token_indices]
                expert_outputs = self.experts[expert_idx](expert_inputs)
                
                # Multiply by routing weights
                weights = topk_weights[token_indices, weight_ranks].unsqueeze(-1)
                out.index_add_(0, token_indices, expert_outputs * weights)
                
        return out.view(*orig_shape)
