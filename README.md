# Picotron

A clean-room, from-scratch LLM pre-training framework inspired by the design principles of Nanotron, but engineered with zero mandatory GPU-specific dependencies. 

## Features

- **Zero Required Hardware Extensions**: Runs out-of-the-box on any PyTorch-compatible CPU or GPU. Doesn't crash on Volta/Turing cards (e.g. V100, T4) due to hard Triton/FlashAttention module-level imports.
- **Compute Capability Aware Precision**: Dynamically queries CUDA hardware capabilities to default to `float16` on Compute Capability < 8.0 (Turing/T4), and `bfloat16` on newer architectures (Ampere/Hopper/Blackwell).
- **Hybrid Attention Backend**: Prefers PyTorch's native `scaled_dot_product_attention` (SDPA) with dynamic runtime checks for `flash_attn` integration if available.
- **ZeRO Stage-1 Optimization**: Distributed training optimizer partitions gradients/optimizer states utilizing pure `torch.distributed` all-reduce patterns.
- **Simple Configuration**: Native python `dataclasses` + `pyyaml` parsing without relying on complex, external deserialization dependencies.
- **Dataset Preprocessing & Run Automation**: Simple YAML-based preprocessor CLI to tokenize datasets and automated boot-scripts.

## Installation

```bash
pip install -e .
```

## Running Training

### Quick Run (Preprocess + Train)
To verify dependencies, download/tokenize your dataset from your YAML config, and launch training (single-GPU or multi-GPU based on config parameters) with a single command:
* **Windows**:
  ```bash
  run_picotrain.bat examples/26M.yaml
  ```
* **Linux / Kaggle**:
  ```bash
  ./run_picotrain.sh examples/26M.yaml
  ```

### Manual Preprocessing
```bash
picotron-preprocess config.yaml
```

### Manual Training
```bash
python train.py config.yaml
```

## Roadmap & TODO List

- [ ] **1. MoE Prep**: Add expert capacity factor optimization, load balancing auxiliary loss terms, and expert parallel routing layouts.
- [x] **2. Make Dataset Prep Easy**: Implement unified command-line tool wrappers to automatically process, filter, and tokenize raw text files directly into `.bin` numpy memories.
