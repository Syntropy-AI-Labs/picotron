"""
Distillation Trainer coordinating student vs. teacher model steps.
Uses forward hooks to align student and teacher hidden layer representations.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from picotron.trainer import Trainer
from picotron.distillation.losses import (
    compute_logit_distillation_loss,
    compute_hidden_state_distillation_loss,
    compute_attention_distillation_loss
)

class ActivationHook:
    """Helper class to capture intermediate module outputs without modifications."""
    def __init__(self):
        self.activation = None
        
    def __call__(self, module, input_module, output_module):
        if isinstance(output_module, tuple):
            self.activation = output_module[0]
        else:
            self.activation = output_module

class DistillationTrainer(Trainer):
    """
    Knowledge Distillation Trainer.
    Aligns student predictions and intermediate layers with a teacher model.
    """
    def __init__(
        self,
        config,
        model: nn.Module,  # Student Model
        teacher_model: nn.Module,
        train_dataloader,
        val_dataloader=None,
        alpha: float = 0.5,      # Coefficient for student standard NLL loss
        beta: float = 0.5,       # Coefficient for logit KL-divergence
        gamma: float = 0.0,      # Coefficient for hidden state distillation
        temperature: float = 2.0,
        student_target_layer: Optional[nn.Module] = None,
        teacher_target_layer: Optional[nn.Module] = None
    ):
        super().__init__(config, model, train_dataloader, val_dataloader)
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        
        # Setup Layer Hooks if hidden state alignment is requested
        self.student_hook = None
        self.teacher_hook = None
        self.projection_layer = None
        
        if self.gamma > 0.0 and student_target_layer is not None and teacher_target_layer is not None:
            self.student_hook = ActivationHook()
            self.teacher_hook = ActivationHook()
            
            student_target_layer.register_forward_hook(self.student_hook)
            teacher_target_layer.register_forward_hook(self.teacher_hook)
            
            # Setup projection layer mapping student dims to teacher dims if they differ
            # We fetch one dummy parameter size
            student_dim = model.config.hidden_size
            teacher_dim = teacher_model.config.hidden_size
            if student_dim != teacher_dim:
                self.projection_layer = nn.Linear(student_dim, teacher_dim, bias=False).to(self.device)
                self.optimizer.add_param_group({"params": self.projection_layer.parameters()})

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Run standard distillation step, combining student NLL and teacher distillation losses."""
        if self.use_deepspeed:
            raise NotImplementedError("DeepSpeed is not supported currently inside DistillationTrainer.")
            
        loss_accum = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        criterion = nn.CrossEntropyLoss()
        
        accum_steps = self.config.train.grad_accum_steps
        for micro_step in range(accum_steps):
            mb_x = x[micro_step * x.size(0) // accum_steps : (micro_step + 1) * x.size(0) // accum_steps]
            mb_y = y[micro_step * y.size(0) // accum_steps : (micro_step + 1) * y.size(0) // accum_steps]
            
            mb_x = mb_x.to(self.device, non_blocking=True)
            mb_y = mb_y.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                # 1. Forward pass Student policy model
                student_logits, aux_loss = self.model(mb_x)
                student_loss_nll = criterion(student_logits.view(-1, student_logits.size(-1)), mb_y.view(-1))
                
                # 2. Forward pass Teacher model (requires no gradients)
                with torch.no_grad():
                    teacher_logits, _ = self.teacher_model(mb_x)
                    
                # 3. Calculate Distillation Logits Loss
                loss_logits = compute_logit_distillation_loss(
                    student_logits,
                    teacher_logits,
                    temperature=self.temperature
                )
                
                # Combined Loss
                loss = (self.alpha * student_loss_nll + self.beta * loss_logits) / accum_steps
                
                # 4. Optional Hidden State Distillation Loss alignment
                if self.gamma > 0.0 and self.student_hook is not None and self.teacher_hook is not None:
                    student_h = self.student_hook.activation
                    teacher_h = self.teacher_hook.activation
                    if student_h is not None and teacher_h is not None:
                        loss_hidden = compute_hidden_state_distillation_loss(
                            student_h,
                            teacher_h,
                            projection_layer=self.projection_layer
                        )
                        loss += (self.gamma * loss_hidden) / accum_steps
                        
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            loss_accum += student_loss_nll.item() / accum_steps

        # Update learning rate
        lr = self.get_lr(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
            
        if self.config.train.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.config.train.grad_clip)
            
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        self.global_step += 1
        return loss_accum
