# Reinforcement Learning with PPO

Build interactive alignment loops utilizing Proximal Policy Optimization (PPO), Actor-Critic parameters, and Reward Models.

---

## 🏛️ PPO Actor-Critic and Reward Layout

PPO updates policies using value function estimations and environmental rewards:

```python
import torch
from picotron.rlhf.ppo import ActorCritic, RewardModel, PPORolloutBuffer, PPOTrainer

# 1. Wrap model elements
actor_critic = ActorCritic(policy_model)
reward_model = RewardModel(reference_model)

# 2. Build optimization engine
optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=1e-5)
ppo_trainer = PPOTrainer(
    actor_critic=actor_critic,
    ref_model=reference_model,
    optimizer=optimizer,
    kl_coef=0.01,
    clip_range=0.2
)
```

---

## 📥 Buffer Experience Rollout Insertion

During policy interactive steps, collect generated sequences and store them inside the rollout buffer:

```python
buffer = PPORolloutBuffer()

# Insert rollout sequence elements
buffer.insert(
    query=query_token_ids,          # [B, PromptLength]
    response=response_token_ids,    # [B, ResponseLength]
    logprob=action_logprobs,        # [B, ResponseLength]
    value=value_predictions,        # [B, ResponseLength]
    reward=reward_scores,           # [B]
    mask=sequence_masks             # [B, ResponseLength]
)
```

---

## 🔄 Optimizing Policy State Parameters

Once the buffer is filled, run policy updates:

```python
# Extract batched experiences
experiences = buffer.get_batches()

# Run a policy gradient training step
ppo_loss = ppo_trainer.train_step_ppo(experiences)
print(f"PPO Optimization Loss: {ppo_loss:.4f}")

# Reset buffer for next rollout iteration
buffer.clear()
```
