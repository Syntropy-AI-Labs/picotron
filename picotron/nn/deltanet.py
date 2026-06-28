"""
Gated DeltaNet (Linear Attention) layer for Qwen 3.5 hybrid architectures.
Implements $O(1)$ recurrent inference state updates and linear memory footprints.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention layer.
    Uses recurrent state updates with input-dependent gating.
    """
    def __init__(self, hidden_size: int, num_heads: int, bias: bool = False):
        """Initialize DeltaNet projections and gates."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Gating projections to control retention and updates
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=bias) # Learning rate / decay gates
        
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using recurrent scan of token sequence."""
        bsz, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        gate = torch.sigmoid(self.gate_proj(hidden_states)).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(hidden_states)).unsqueeze(-1).transpose(1, 2) # [bsz, heads, seq_len, 1, 1]
        
        # Initialize hidden state: [bsz, heads, head_dim, head_dim]
        # We loop sequentially for training simplicity (can be accelerated via custom scans)
        S = torch.zeros(bsz, self.num_heads, self.head_dim, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        outputs = []
        
        for t in range(seq_len):
            q_t = q[:, :, t].unsqueeze(-1)  # [bsz, heads, head_dim, 1]
            k_t = k[:, :, t].unsqueeze(-2)  # [bsz, heads, 1, head_dim]
            v_t = v[:, :, t].unsqueeze(-1)  # [bsz, heads, head_dim, 1]
            
            beta_t = beta[:, :, t]  # [bsz, heads, 1, 1]
            gate_t = gate[:, :, t]  # [bsz, heads, head_dim]
            
            # Compute delta update: S_t = S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t^T) @ k_t
            # Representing the delta rule update
            pred_v = torch.matmul(S, k_t.transpose(-2, -1)) # [bsz, heads, head_dim, 1]
            error = v_t - pred_v
            
            # Update state S
            S = S + beta_t * torch.matmul(error, k_t)
            
            # Retrieve output: y_t = (S_t @ q_t) * gate_t
            y_t = torch.matmul(S, q_t).squeeze(-1) # [bsz, heads, head_dim]
            y_t = y_t * gate_t
            outputs.append(y_t.unsqueeze(2)) # Store output token
            
        out = torch.cat(outputs, dim=2) # [bsz, heads, seq_len, head_dim]
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(out)
