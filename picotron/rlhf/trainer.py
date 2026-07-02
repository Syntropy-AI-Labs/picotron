"""
Preference Tuning Trainer (DPO, ORPO, GRPO).
Orchestrates log prob evaluation on policy vs. reference model.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from picotron.trainer import Trainer
from picotron.rlhf.loss import compute_dpo_loss, compute_orpo_loss

class PreferenceTrainer(Trainer):
    """
    Trainer for Alignment Preference Tuning.
    Supports DPO (Direct Preference Optimization) and ORPO (Odds Ratio Preference Optimization).
    """
    def __init__(self, config, model: nn.Module, ref_model: Optional[nn.Module], train_dataloader, val_dataloader=None, mode: str = "dpo", beta: float = 0.1):
        super().__init__(config, model, train_dataloader, val_dataloader)
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
            # Reference model does not require gradients
            for param in self.ref_model.parameters():
                param.requires_grad = False
                
        self.mode = mode.lower()
        self.beta = beta

    def get_log_probs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Extract log probability sums of label targets from logits, ignoring -100 masks."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        loss_mask = shift_labels != -100
        
        # Clamp labels to gather without index out of bounds
        clamped_labels = shift_labels.clone()
        clamped_labels[shift_labels == -100] = 0
        
        per_token_logps = torch.gather(log_probs, dim=-1, index=clamped_labels.unsqueeze(-1)).squeeze(-1)
        return (per_token_logps * loss_mask).sum(dim=-1)

    def train_step(self, x, y=None) -> float:
        if isinstance(x, dict):
            return self.train_step_preference(x)
        return super().train_step(x, y)

    def train_step_preference(self, batch: dict) -> float:
        """Compute training losses on policy vs reference outputs."""
        self.optimizer.zero_grad(set_to_none=True)
        
        chosen_input = batch["chosen_input"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_input = batch["rejected_input"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            # 1. Forward passes through Policy model
            chosen_logits, _ = self.model(chosen_input)
            rejected_logits, _ = self.model(rejected_input)
            
            policy_chosen_logps = self.get_log_probs(chosen_logits, chosen_labels)
            policy_rejected_logps = self.get_log_probs(rejected_logits, rejected_labels)
            
            if self.mode == "dpo":
                # DPO requires reference model forward passes
                with torch.no_grad():
                    ref_chosen_logits, _ = self.ref_model(chosen_input)
                    ref_rejected_logits, _ = self.ref_model(rejected_input)
                    ref_chosen_logps = self.get_log_probs(ref_chosen_logits, chosen_labels)
                    ref_rejected_logps = self.get_log_probs(ref_rejected_logits, rejected_labels)
                    
                loss, _, _ = compute_dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                    beta=self.beta
                )
            elif self.mode == "orpo":
                # ORPO calculates log odds ratios directly
                # Compute Chosen Net Log Likelihood (NLL loss)
                shift_logits = chosen_logits[..., :-1, :].contiguous()
                shift_labels = chosen_labels[..., 1:].contiguous()
                nll_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss = compute_orpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    policy_chosen_nll=nll_loss,
                    beta=self.beta
                )
            else:
                raise ValueError(f"Unsupported preference mode: {self.mode}")

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
            
        self.global_step += 1
        return loss.item()
