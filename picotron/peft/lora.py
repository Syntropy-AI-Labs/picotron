"""
Implementation of LoRA (Low-Rank Adaptation) and DoRA (Weight-Decomposed Low-Rank Adaptation) layers.
Supports custom base weight quantizations for QLoRA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer wrapping a standard Linear projection.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        use_dora: bool = False
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.use_dora = use_dora
        
        # Disable gradients on base layers
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        self.dropout = nn.Dropout(p=lora_dropout)
        
        # DoRA magnitude parameters
        if self.use_dora:
            # Directional weight vector scale magnitude m = ||W||
            weight = self.base_layer.weight.data.float()
            self.m = nn.Parameter(torch.norm(weight, p=2, dim=1, keepdim=True))
            
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        """Initialize low-rank matrices (Kaiming uniform for A, zero for B)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merge low-rank adapter weights into base layer."""
        if self.merged:
            return
        if self.use_dora:
            # DoRA weight merging
            weight = self.base_layer.weight.data.float()
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            fused_weight = weight + lora_weight
            norm = torch.norm(fused_weight, p=2, dim=1, keepdim=True)
            self.base_layer.weight.data = (self.m * (fused_weight / norm)).to(self.base_layer.weight.dtype)
        else:
            # Standard LoRA merging
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += lora_weight.to(self.base_layer.weight.dtype)
        self.merged = True

    def unmerge(self):
        """Extract merged adapter weights from base layer."""
        if not self.merged:
            return
        if self.use_dora:
            # Reconstruct original base weights using magnitude
            weight = self.base_layer.weight.data.float()
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data = (weight - lora_weight).to(self.base_layer.weight.dtype)
        else:
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= lora_weight.to(self.base_layer.weight.dtype)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)
            
        base_out = self.base_layer(x)
        
        if self.use_dora:
            # DoRA forward: y = x @ (m * (W + lora) / ||W + lora||)
            weight = self.base_layer.weight.data.float()
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            fused_weight = weight + lora_weight
            norm = torch.norm(fused_weight, p=2, dim=1, keepdim=True)
            dora_weight = self.m * (fused_weight / norm)
            
            # Perform projection using calculated dora weights
            return F.linear(x, dora_weight.to(x.dtype), self.base_layer.bias)
            
        # Standard LoRA forward: y = W_x + (x @ A.T @ B.T) * scaling
        adapter_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + adapter_out
