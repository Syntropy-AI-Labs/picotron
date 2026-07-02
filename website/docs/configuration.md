# Model Architecture Configuration

Picotron features native configuration profile support for popular architectures including GPT, LLaMA, Qwen, Gemma, DeepSeek, Mistral, and custom Mixture of Experts (MoE).

---

## 🛠️ Structuring a Configuration Profile (`config.yaml`)

Configuration profiles partition parameters into model architecture, parallel layouts, datasets, and optimization cycles:

```yaml
model:
  vocab_size: 50257                # Size of token dictionary
  hidden_size: 512                 # Latent hidden dimension size
  num_hidden_layers: 8             # Number of Transformer blocks
  num_attention_heads: 8           # Attention heads count
  num_key_value_heads: 4           # K/V heads count (Grouped Query Attention)
  intermediate_size: 1024          # Feed-Forward expansion dimension
  max_position_embeddings: 1024    # Maximum sequence length context window
  norm_type: "rms"                 # Layer normalization type ('rms' or 'layer')
  activation_type: "silu"          # Activation choice ('silu', 'gelu', etc.)
  rms_norm_eps: 0.000005           # Epsilon stabilizer parameter
  bias: false                      # Enable linear bias weight vectors
  use_moe: false                   # Enable Mixture of Experts (MoE) block

parallel:
  dp_size: 1                       # Data Parallelism shard size
  zero_stage: 0                    # Zero Redundancy Stage (0 = disabled, 1 = stage 1)

data:
  dataset_path: "data/tokens.bin"  # Path to preprocessed train binaries
  sequence_length: 1024            # Active training token context length
  micro_batch_size: 8              # Gradient calculation step batch size
  num_workers: 8                   # Prefetch worker count

train:
  learning_rate: 0.0006            # Peak learning rate parameter
  min_learning_rate: 0.00006       # Cosine scheduler floor parameter
  weight_decay: 0.1                # AdamW weight regularization
  max_steps: 1000                  # Maximum iteration steps
  warmup_steps: 100                # Cosine warmup segment steps
  grad_accum_steps: 4              # Gradient accumulation intervals
  seed: 42                         # Randomized initialization seed
  compile: true                    # PyTorch 2.0 graph compilation
  use_cuda_graphs: false           # Static graph caching acceleration
  mixed_precision: "bf16"          # Execution precision ('fp32', 'fp16', 'bf16')
  save_checkpoint: true            # Enable periodic checkpointing saves
  checkpoint_dir: "checkpoints/"   # Target directory for checkpoint outputs
```

---

## 🚀 Loading Configurations in Python

To parse config files in your training pipelines:

```python
from picotron.config import load_config_from_yaml

# Load configuration profile
config = load_config_from_yaml("config.yaml")

print(f"Loaded vocab size: {config.model.vocab_size}")
print(f"Loaded learning rate: {config.train.learning_rate}")
```
