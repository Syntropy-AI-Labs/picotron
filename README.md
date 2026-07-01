# Picotron ⚡

A clean-room, high-performance LLM training and optimization framework. Engineered to scale efficiently from lightweight local research setups to multi-GPU clusters.

---

## 🚀 Key Framework Pillars

### 1. ⚡ High-Performance Triton Kernels (`picotron/kernels/triton_kernels/`)
To bypass PyTorch framework overhead and minimize GPU memory roundtrips, Picotron includes hand-optimized Triton kernels:
* **Fused RMSNorm & LayerNorm**: Normalization reductions and scaling executed inside a single GPU pass.
* **Fused SwiGLU & GeGLU**: Computes activation projections directly in local SRAM.
* **Fused RoPE (Rotary Embeddings)**: Performs position vector rotation directly inside the GPU register file.
* **Fused Cross-Entropy**: Fuses logit projections, stable logsumexp reduction, and gradient accumulation.
* **Fused AdamW & SGD**: Updates moments and applies weight decays in a single optimized pass.

*Note: All kernels feature automated fallback checks. If Triton is not installed, the framework seamlessly executes standard PyTorch code.*

### 2. 🧠 Modern Sequence Mixers (`picotron/nn/mixers.py`)
Picotron supports cutting-edge sequence modeling components out of the box:
* **Multi-Head Latent Attention (MLA)**: DeepSeek-style compressed KV cache projections to sustain large context sizes.
* **Selective SSM (Mamba-style)**: Input-dependent state discretization transitions for $O(N)$ sequence scaling.
* **RWKV Recurrence**: Linear recurrence decay matrices for recurrent state formulations.
* **Gated DeltaNet**: Pure linear attention recurrences with input-dependent gates.

### 3. 🎛️ PEFT & SFT Fine-Tuning (`picotron/peft/`)
Unified Parameter-Efficient Fine-Tuning (PEFT) APIs to adapt pre-trained weights:
* **LoRA (Low-Rank Adaptation)**: Injects trainable rank $r$ parameters into targeted projection weight coordinates.
* **DoRA (Weight-Decomposed Low-Rank Adaptation)**: Decomposes target linear weights into magnitude and directional scaling vectors.
* **SFT Trainer**: Implements response-masked instruction fine-tuning, masking out prompt labels using `-100` values.
* **Merge & Unmerge**: Fuses adapter parameter layers back into base weight matrices for zero-overhead inference.

### 4. 🔗 Preference Alignment RLHF (`picotron/rlhf/`)
Allows fine-tuning models based on preference dataset pairs:
* **DPO (Direct Preference Optimization)**: Optimizes policies against a frozen reference model using preference log-ratios.
* **ORPO (Odds Ratio Preference Optimization)**: Combines supervised instruction tuning and Odds-Ratio penalties without requiring a reference model.
* **GRPO**: Group Relative Policy Optimization for efficient policy advantage estimations.
* **PPO Trainer**: Implements Actor-Critic, Generalized Advantage Estimation (GAE), KL penalties, and rollout buffers.

### 5. 🎓 Knowledge Distillation (`picotron/distillation/`)
Facilitates training lightweight student models from large teachers:
* **Logit Distillation**: Kullback-Leibler divergence matching with temperature scaling.
* **Hidden State Distillation**: Matches intermediate representation states using Mean Squared Error (MSE), applying linear projections if dimensions differ.
* **Activation Hooks**: Non-intrusive forward hooks to capture hidden layers during model execution.

---

## 🛠️ Performance Accelerations
* **Asynchronous Dataloader Prefetching**: Overlaps CPU-to-GPU tensor memory transfers with GPU model execution.
* **Async Background Checkpoints**: Moves state weights to CPU memory asynchronously and delegates disk writing to a background thread to prevent step execution pauses.
* **CUDA Graphs**: Captures static training steps inside a CUDA execution graph, bypassing CPU-side kernel launch latencies.
* **Memory Pooling**: Configures the CUDA caching allocator limits to pool allocations and prevent fragmentations.
