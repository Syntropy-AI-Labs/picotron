"""
Distillation loss functions for knowledge transfer.
Supports logit KL-divergence, hidden state projections, attention map alignments, and temperature scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def compute_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0
) -> torch.Tensor:
    """
    Kullback-Leibler (KL) divergence distillation loss on student and teacher logits.
    Applies temperature scaling to soften probability distributions.
    """
    # Scale student/teacher log probabilities by temperature
    p_student = F.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL-Divergence loss multiplied by T^2 to scale gradients appropriately
    kl_loss = F.kl_div(p_student, p_teacher, reduction="batchmean") * (temperature ** 2)
    return kl_loss

def compute_hidden_state_distillation_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projection_layer: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Hidden state alignment loss using Mean Squared Error (MSE).
    If student hidden size differs from teacher hidden size, applies a projection layer.
    """
    if projection_layer is not None:
        student_hidden = projection_layer(student_hidden)
        
    return F.mse_loss(student_hidden.float(), teacher_hidden.float())

def compute_attention_distillation_loss(
    student_attention_probs: torch.Tensor,
    teacher_attention_probs: torch.Tensor
) -> torch.Tensor:
    """
    Aligns attention maps by calculating Mean Squared Error (MSE) on self-attention probability matrices.
    """
    return F.mse_loss(student_attention_probs.float(), teacher_attention_probs.float())
