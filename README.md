# Picotron

A clean-room, from-scratch LLM pre-training framework inspired by the design principles of Nanotron, but engineered with zero mandatory GPU-specific dependencies. 

Picotron enables pre-training of modern language models—including hybrid architectures—on anything from standard local CPUs to multi-GPU clusters, completely avoiding "dependency hell" from hardware-specific packages like FlashAttention, Triton, or custom CUDA extensions.

---

## Features

- **Zero Required Hardware Extensions**: Runs out-of-the-box on any PyTorch-compatible CPU or GPU. Doesn't crash on Volta/Turing cards (e.g., V100, T4) due to hard Triton/FlashAttention module-level imports.
- **Compute Capability Aware Precision**: Dynamically queries CUDA hardware capabilities to default to `float16` on Compute Capability < 8.0 (Turing/T4), and `bfloat16` on newer architectures (Ampere/Hopper/Blackwell).
- **Hybrid Attention Backend**: Prefers PyTorch's native `scaled_dot_product_attention` (SDPA) with dynamic runtime checks for `flash_attn` integration if available.
- **ZeRO Stage-1 Optimization**: Distributed training optimizer partitions gradients/optimizer states utilizing pure `torch.distributed` all-reduce patterns.
- **Simple Configuration**: Native python `dataclasses` + `pyyaml` parsing without relying on complex, external deserialization dependencies.
- **Dataset Preprocessing & Run Automation**: Simple YAML-based preprocessor CLI to tokenize datasets and automated boot-scripts.

---

## Supported Architectures

By configuring the parameters inside your `config.yaml`, Picotron can emulate a wide variety of state-of-the-art architectures:

### 1. LLaMA 3.1
* Uses Grouped Query Attention (GQA), SwiGLU activations, and RMSNorm pre-normalization.
* Config: Set `norm_type: "rms"`, `activation_type: "silu"`, and adjust your `rope_theta` (typically `500000.0` for LLaMA 3).

### 2. Gemma 2
* Interleaves global attention and local sliding window attention.
* Config: Set `scale_embeddings: true` (scales embedding vectors by $\sqrt{d_{model}}$), `activation_type: "gelu"` (for GeGLU), `alternate_sliding_window: true`, and adjust `sliding_window` size (typically `4096`). Supports logit soft-capping using `logit_soft_cap`.

### 3. Qwen 2.5
* Focuses on dense decoders with GQA, SwiGLU, and QK-Norm (stabilizes queries/keys by applying RMSNorm before compute).
* Config: Set `qk_norm: true`, `activation_type: "silu"`, and `norm_type: "rms"`.

### 4. Qwen 3.5 (Hybrid Gated DeltaNet)
* Implements a hybrid sequence mixer that alternates between standard softmax attention and linear recurrence layers.
* Config: Set `use_deltanet: true` and `deltanet_ratio: 3` (runs 3 Gated DeltaNet layers for every 1 standard Attention layer).

### 5. StarCoder 2
* Traditional code-optimized architecture that uses standard LayerNorm instead of RMSNorm and exposes biases on projections.
* Config: Set `norm_type: "layer"`, `bias: true`, and `activation_type: "silu"`.

---

## Advanced Components

### Mixture of Experts (MoE)
Picotron includes a native **Mixture of Experts** layer to replace standard MLPs. Tokens are routed to the top-k experts using a linear gating network:
* **Load Balancing Loss**: Includes a built-in auxiliary loss function ($\mathcal{L}_{aux} = N \sum_{i=1}^N P_i \cdot F_i$) to prevent expert collapse (i.e., routing all tokens to the same expert).
* Config parameters:
  ```yaml
  model:
    use_moe: true
    moe_num_experts: 8
    moe_top_k: 2
  ```

### Gated DeltaNet (Linear Attention)
DeltaNet replaces quadratic attention with $O(1)$ linear recurrent updates, acting as a stable and memory-efficient alternative for processing long sequences. The state updates are governed by input-dependent learning rate gates ($\beta_t$):
$$S_t = S_{t-1} + \beta_t \cdot (v_t - S_{t-1} k_t) \otimes k_t$$

---

## Installation

To install Picotron and register the command-line helper scripts:
```bash
pip install -e .
```

---

## Running Preprocessing & Training

### 1. Simple CLI Execution (Preprocess + Train)
Automatically download/tokenize datasets declared in your configuration and launch training. The script automatically determines whether to run single-device or distributed multi-device training (using `torchrun` NCCL/Gloo backends) based on `parallel.dp_size` in your config:
* **Windows**:
  ```bash
  run_picotrain.bat examples/26M.yaml
  ```
* **Linux / Kaggle**:
  ```bash
  ./run_picotrain.sh examples/26M.yaml
  ```

### 2. Manual Commands
If you prefer running the pipeline phases manually:
* **Tokenize Datasets**:
  ```bash
  picotron-preprocess config.yaml
  ```
* **Launch Pretraining**:
  ```bash
  python train.py config.yaml
  ```

---

## Roadmap & TODO List

- [x] **1. MoE Prep**: Add expert capacity factor optimization, load balancing auxiliary loss terms, and expert parallel routing layouts.
- [x] **2. Make Dataset Prep Easy**: Implement unified command-line tool wrappers to automatically process, filter, and tokenize raw text files directly into `.bin` numpy memories.
