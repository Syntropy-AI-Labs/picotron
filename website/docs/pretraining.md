# Pre-Training Pipeline

Orchestrate data cleaning, async tokenization, sequence packing, sharding, and core foundation training.

---

## 📦 Phase 1: High-Performance Data Preprocessing

Picotron packages an all-in-one preprocessing console entrypoint `picotron-data`. This tool cleans, tokenizes, shards, and packs sequence data using multi-core CPU async pipelines:

### Single Preprocessing Command
```bash
picotron-data \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --tokenizer gpt2 \
  --output_dir data/ \
  --seq_len 1024 \
  --num_workers 8
```

This generates `data/foundation_tokens.bin` and index cache mappings configured for fast memory-mapped indexing access.

---

## 🏋️ Phase 2: Launching Pre-Training

To run the pre-training loop over the preprocessed datasets:

```bash
picotron-train config.yaml
```

---

## ⚡ PyTorch Performance Accelerations

Picotron utilizes several optimization structures under the hood:

### 1. Asynchronous CUDA Prefetch Dataloader
Hides data transfer latency by transferring the next batch to GPU memory in a background stream:

```python
from picotron.trainer import PrefetchDataloader

# Prefetch dataloader wraps standard PyTorch DataLoader instances
prefetch_loader = PrefetchDataloader(standard_dataloader, device="cuda")
```

### 2. Selective Activation Recomputation
Reduces activation memory footprint by only recomputing attention maps during backward steps:

```python
from picotron.nn.recompute import checkpoint

# Wrap attention block execution
attention_output = checkpoint(attention_layer, input_tensor)
```
