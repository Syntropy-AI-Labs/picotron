"""
Expose Knowledge Distillation trainer and loss utilities.
"""

from picotron.distillation.losses import (
    compute_logit_distillation_loss,
    compute_hidden_state_distillation_loss,
    compute_attention_distillation_loss
)
from picotron.distillation.trainer import DistillationTrainer
