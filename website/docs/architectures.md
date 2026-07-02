# Model Architectures

Picotron supports a diverse set of sequence mixers and model configurations. Configure architectures using the standard YAML config profile settings.

---

## 🏛️ LLaMA, Mistral, & Qwen
These architectures use Grouped-Query Attention (GQA) and SwiGLU Feed-Forward Networks (FFN):

* **LLaMA Config Parameters**:
  ```yaml
  model:
    vocab_size: 32000
    hidden_size: 4096
    num_hidden_layers: 32
    num_attention_heads: 32
    num_key_value_heads: 8    # GQA ratio 4:1
    intermediate_size: 11008
    norm_type: "rms"
    activation_type: "silu"
  ```

---

## ⚡ Multi-Head Latent Attention (MLA)
DeepSeek V2 and V3 models utilize MLA to compress the Key-Value (KV) cache size down to a fraction of standard GQA size. This compression is achieved through low-rank projection while maintaining RoPE positioning signals.

* **MLA Config Parameters**:
  ```yaml
  model:
    vocab_size: 102400
    hidden_size: 5120
    num_hidden_layers: 64
    num_attention_heads: 128
    kv_lora_rank: 512          # Latent KV compression bottleneck
    q_lora_rank: 128           # Latent Query compression bottleneck
    rope_dim: 64               # Decoupled RoPE coordinates size
  ```

---

## 🌀 State Space Models (Mamba-style SSM)
Mamba architectures replace standard attention matrices with dynamic Selective SSM transitions, enabling $O(N)$ linear sequence context length scaling.

* **SSM Config Parameters**:
  ```yaml
  model:
    vocab_size: 50257
    hidden_size: 2048
    num_hidden_layers: 48
    activation_type: "silu"
    d_state: 16                # Latent SSM state dimension size
    d_inner: 4096              # Expanded inner dimension size
  ```
