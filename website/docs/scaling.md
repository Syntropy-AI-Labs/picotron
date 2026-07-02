# Scaling & Distributed Parallelism

Picotron implements advanced distributed training paradigms (TP, PP, CP, EP) to support massive model training across large cluster nodes.

---

## ⛓️ 3D Parallelism Config Layouts

Structure parallel configurations to partition memory weights, activations, and routing configurations:

```yaml
parallel:
  tp_size: 2        # Tensor Parallel size (split linear layers within a node)
  pp_size: 2        # Pipeline Parallel size (split layers across nodes)
  dp_size: 2        # Data Parallel size (sharded batches)
  cp_size: 1        # Context Parallel size (shard along temporal dimension)
  ep_size: 2        # Expert Parallel size (split mixture experts across nodes)
  zero_stage: 1     # Zero Redundancy Stage (Stage 1 Optimizer State Sharding)
```

---

## 1. Tensor Parallelism (TP)
Megatron-LM style tensor parallelism splits linear operations:
* **ColumnParallelLinear**: Splits projections ($W_Q, W_K, W_V$ or MLP Gate/Up weights) along columns.
* **RowParallelLinear**: Splits outputs ($W_O$ or MLP Down weights) along rows, applying a distributed `all_reduce` step to sum the gathered outputs.

---

## 2. Pipeline Parallelism (PP)
Partitions model layers sequentially across devices. Activations and gradients are handed off between consecutive stages:
* **Forward Pass**: Stage $N$ receives activations from Stage $N-1$, runs forward evaluation, and sends them to Stage $N+1$.
* **Backward Pass**: Stage $N$ receives loss gradients from Stage $N+1$, runs backward evaluation, and sends gradients to Stage $N-1$.

---

## 3. Context Parallelism (CP)
Shards long sequence tensors along the temporal (sequence length) dimension. Useful for extreme context length tasks that exceed the memory limits of a single accelerator device.

---

## 4. Expert Parallelism (EP)
Distributes individual experts of a Mixture-of-Experts (MoE) layer across GPUs. Token routing coordinates use high-speed PyTorch `all_to_all_single` calls to route tokens to their respective experts.
