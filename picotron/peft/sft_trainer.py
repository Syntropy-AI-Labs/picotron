"""
Supervised Fine-Tuning (SFT) Trainer.
Supports instruction-tuning loss calculation (masking out prompt tokens so loss is computed only on responses).
"""

import os
import torch
import torch.nn as nn
from picotron.trainer import Trainer
from picotron.kernels.triton_kernels import triton_cross_entropy

class SFTTrainer(Trainer):
    """
    Specialized trainer for Supervised Fine-Tuning (SFT) on instructions.
    Only computes cross-entropy loss on response tokens, masking out prompts using a special label padding value (-100).
    """
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Run single SFT training step, calculating loss only on response tokens."""
        if self.use_deepspeed:
            raise NotImplementedError("DeepSpeed is not supported currently inside SFTTrainer fallback.")
            
        loss_accum = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        
        accum_steps = self.config.train.grad_accum_steps
        for micro_step in range(accum_steps):
            mb_x = x[micro_step * x.size(0) // accum_steps : (micro_step + 1) * x.size(0) // accum_steps]
            mb_y = y[micro_step * y.size(0) // accum_steps : (micro_step + 1) * y.size(0) // accum_steps]
            
            mb_x = mb_x.to(self.device, non_blocking=True)
            mb_y = mb_y.to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                logits, aux_loss = self.model(mb_x)
                
                # Reshape for cross entropy
                logits_2d = logits.view(-1, logits.size(-1))
                targets_1d = mb_y.view(-1)
                
                # Filter out tokens marked with -100 (prompts / padding) to compute loss only on responses
                valid_mask = targets_1d != -100
                if valid_mask.sum() > 0:
                    loss_ce = triton_cross_entropy(logits_2d[valid_mask], targets_1d[valid_mask])
                else:
                    loss_ce = torch.tensor(0.0, device=self.device, requires_grad=True)
                    
                loss = (loss_ce + 0.01 * aux_loss) / accum_steps
                
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            loss_accum += loss_ce.item() / accum_steps

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
