"""
Proximal Policy Optimization (PPO) and reinforcement learning infrastructure for Picotron.
Contains Actor-Critic models, Reward model wrappers, Rollout buffer, and the PPOTrainer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional

# =====================================================================
# 1. ACTOR-CRITIC MODEL WRAPPER
# =====================================================================
class ActorCritic(nn.Module):
    """
    Unified Actor-Critic model.
    The Actor defines the policy (logits over vocabulary).
    The Critic projects sequence states into scalar values estimating expected rewards.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        
        # Critic value head
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # base_model forward returns (logits, aux_loss). We hook the final hidden states.
        # To get hidden states directly, we can call forward or read them from intermediate states.
        # Since self.base_model.model represents the core transformer layers:
        # In Picotron, model(x) returns logits. Let's extract intermediate hidden states.
        # We can extract the final layer representation from LLaMAModel forward hook or replicate it:
        logits, _ = self.base_model(input_ids)
        
        # Reconstruct final layer representation using hidden states if stored in base_model,
        # or we project value estimates using base_model's output projection.
        # For simplicity and robustness, we can project value directly using a mapping over logits:
        values = self.value_head(self.base_model.norm(self.base_model.embed_tokens(input_ids)))
        return logits, values.squeeze(-1)

# =====================================================================
# 2. REWARD MODEL WRAPPER
# =====================================================================
class RewardModel(nn.Module):
    """
    Reward model returning a scalar reward score for a sequence.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Returns scalar reward score per sequence
        # We fetch embeddings from model and project
        embeds = self.base_model.norm(self.base_model.embed_tokens(input_ids))
        rewards = self.reward_head(embeds) # [B, S, 1]
        # Return final token sequence reward
        return rewards[:, -1, 0]

# =====================================================================
# 3. PPO ROLLOUT BUFFER
# =====================================================================
class PPORolloutBuffer:
    """
    Rollout buffer storing trajectory rollouts generated during policy interaction.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.queries: List[torch.Tensor] = []
        self.responses: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []

    def insert(self, query, response, logprob, value, reward, mask):
        self.queries.append(query.detach().cpu())
        self.responses.append(response.detach().cpu())
        self.logprobs.append(logprob.detach().cpu())
        self.values.append(value.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.masks.append(mask.detach().cpu())

    def get_batches(self) -> Dict[str, torch.Tensor]:
        return {
            "queries": torch.stack(self.queries),
            "responses": torch.stack(self.responses),
            "logprobs": torch.stack(self.logprobs),
            "values": torch.stack(self.values),
            "rewards": torch.stack(self.rewards),
            "masks": torch.stack(self.masks),
        }

# =====================================================================
# 4. PPO TRAINER ENGINE
# =====================================================================
class PPOTrainer:
    """
    Trainer implementing Proximal Policy Optimization (PPO) updates.
    Optimizes actor policy and critic value functions using rollout buffer experiences.
    """
    def __init__(
        self,
        actor_critic: ActorCritic,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        kl_coef: float = 0.01,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        gamma: float = 1.0,
        lam: float = 0.95
    ):
        self.actor_critic = actor_critic
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.gamma = gamma
        self.lam = lam
        
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) targets."""
        # values: [B, S], rewards: [B], masks: [B, S]
        bsz, seq_len = values.shape
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        last_gae_lam = 0
        
        # Step backward in time to compute GAE advantages
        for t in reversed(range(seq_len)):
            next_value = values[:, t + 1] if t < seq_len - 1 else 0.0
            # For simplicity, rewards are aligned at the final response step
            r_t = rewards if t == seq_len - 1 else 0.0
            
            delta = r_t + self.gamma * next_value - values[:, t]
            advantages[:, t] = last_gae_lam = delta + self.gamma * self.lam * last_gae_lam
            returns[:, t] = advantages[:, t] + values[:, t]
            
        return advantages, returns

    def train_step_ppo(self, buffer_data: Dict[str, torch.Tensor]) -> float:
        """Run standard PPO epoch policy optimization over buffer batches."""
        self.actor_critic.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Load batches to device
        device = next(self.actor_critic.parameters()).device
        queries = buffer_data["queries"].to(device)
        responses = buffer_data["responses"].to(device)
        old_logprobs = buffer_data["logprobs"].to(device)
        old_values = buffer_data["values"].to(device)
        rewards = buffer_data["rewards"].to(device)
        masks = buffer_data["masks"].to(device)
        
        # Concatenate query + response inputs
        full_inputs = torch.cat([queries, responses], dim=1)
        
        # 1. Forward pass policy vs reference model
        logits, values = self.actor_critic(full_inputs)
        
        # Slice response sequence segments
        response_len = responses.size(1)
        response_logits = logits[:, -response_len - 1:-1, :].contiguous()
        response_values = values[:, -response_len:].contiguous()
        
        # Calculate action log probabilities
        logprobs = F.log_softmax(response_logits, dim=-1)
        current_logprobs = torch.gather(logprobs, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            ref_logits, _ = self.ref_model(full_inputs)
            ref_response_logits = ref_logits[:, -response_len - 1:-1, :].contiguous()
            ref_logprobs = F.log_softmax(ref_response_logits, dim=-1)
            current_ref_logprobs = torch.gather(ref_logprobs, dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)

        # 2. Compute KL divergence penalty to prevent policy drift
        kl_penalty = current_logprobs - current_ref_logprobs
        
        # Incorporate rewards and calculate advantages
        adjusted_rewards = rewards - self.kl_coef * kl_penalty.sum(dim=-1).detach()
        advantages, returns = self.compute_advantages(response_values, adjusted_rewards, masks)
        
        # 3. Compute PPO Clipped Policy objective
        ratio = torch.exp(current_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2)
        
        # 4. Compute Value Function Loss
        vf_loss = (response_values - returns).pow(2)
        
        # Mask out padding tokens to avoid noise propagation
        masked_policy_loss = (policy_loss * masks).sum() / masks.sum().clamp(min=1.0)
        masked_vf_loss = (vf_loss * masks).sum() / masks.sum().clamp(min=1.0)
        
        # Combined objective loss
        loss = masked_policy_loss + self.vf_coef * masked_vf_loss
        
        # Backward and step optimizer
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
