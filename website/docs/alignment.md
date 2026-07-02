# Preference Alignment & Tuning

Align pre-trained models with human preferences using DPO (Direct Preference Optimization), ORPO (Odds Ratio Preference Optimization), and GRPO (Group Relative Policy Optimization) without reinforcement learning environments.

---

## 🎭 Direct Preference Optimization (DPO)

DPO optimizes policies by comparing action ratios directly against a frozen reference model:

```python
from picotron.rlhf.trainer import PreferenceTrainer

dpo_trainer = PreferenceTrainer(
    config=config,
    model=policy_model,
    ref_model=reference_model,
    train_dataloader=preference_dataloader,
    mode="dpo",
    beta=0.1
)

# Run optimization step
dpo_loss = dpo_trainer.train_step(preference_batch)
```

---

## ⚖️ Odds Ratio Preference Optimization (ORPO)

ORPO combines Supervised Fine-Tuning (SFT) and log odds ratio optimization directly on chosen vs rejected outputs, removing the need for a secondary reference model to save GPU memory:

```python
orpo_trainer = PreferenceTrainer(
    config=config,
    model=policy_model,
    ref_model=None,  # No reference model needed!
    train_dataloader=preference_dataloader,
    mode="orpo",
    beta=0.1
)

# Run optimization step
orpo_loss = orpo_trainer.train_step(preference_batch)
```

---

## 👥 Group Relative Policy Optimization (GRPO)

GRPO normalizes outputs across groups, estimating sample advantages relative to the group mean and standard deviation:

```python
grpo_trainer = PreferenceTrainer(
    config=config,
    model=policy_model,
    ref_model=reference_model,
    train_dataloader=preference_dataloader,
    mode="grpo",
    beta=0.1
)

# Run optimization step
grpo_loss = grpo_trainer.train_step(preference_batch)
```
