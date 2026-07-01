"""
Configuration schemas and loaders for Picotron.
Uses standard library dataclasses and PyYAML for dacite-free deserialization.
"""

import yaml
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, Dict, Any, Type, TypeVar, List

T = TypeVar("T")

@dataclass
class ModelConfig:
    """Configuration options for LLaMA model architecture."""
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    
    # Modern architectures additions
    qk_norm: bool = False  # Normalize Query and Key tensors before dot product
    use_mla: bool = False  # Multi-head Latent Attention
    mla_kv_lora_rank: int = 512
    mla_qk_lora_rank: int = 128
    mla_qk_rope_lora_rank: int = 64
    sliding_window: Optional[int] = None  # Sliding window attention
    
    # Position embedding type: "rope", "nope", "yarn"
    position_embedding_type: str = "rope"
    rope_scaling_factor: float = 1.0  # YaRN scaling factor
    
    # FFN options
    use_moe: bool = False  # Mixture of Experts
    moe_num_experts: int = 8
    moe_top_k: int = 2
    
    # Parallel Attention and FFN Execution
    parallel_attn_ffn: bool = False
    
    # Soft capping of logits (Gemma 2 style)
    logit_soft_cap: Optional[float] = None
    
    # Gemma, StarCoder, GLM & OLMo additions
    bias: bool = False  # Toggle bias on all linear layers
    norm_type: str = "rms"  # "rms" (RMSNorm) or "layer" (LayerNorm)
    alternate_sliding_window: bool = False  # Interleave sliding window attention blocks
    scale_embeddings: bool = False  # Scale input embeddings by sqrt(hidden_size)
    activation_type: str = "silu"  # "silu" (SwiGLU) or "gelu" (GeGLU)
    
    # Qwen 3.5 Gated DeltaNet additions
    use_deltanet: bool = False  # Enable hybrid Attention + Gated DeltaNet architecture
    deltanet_ratio: int = 3  # Ratio of DeltaNet layers to standard Attention layers (e.g. 3:1)

@dataclass
class ParallelConfig:
    """Configuration for distributed parallelism."""
    dp_size: int = 1
    zero_stage: int = 0  # 0: disabled, 1: optimizer state partitioning

@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_path: str = ""
    validation_dataset_path: Optional[str] = None # Path to validation binary data
    sequence_length: int = 2048
    micro_batch_size: int = 4
    num_workers: int = 2

@dataclass
class TrainConfig:
    """Configuration for training process and optimizer."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    max_steps: int = 1000
    warmup_steps: int = 100
    lr_decay_steps: Optional[int] = None
    min_learning_rate: float = 3e-5
    compile: bool = False
    mixed_precision: str = "auto"  # "auto", "fp16", "bf16", "fp32"
    seed: int = 42
    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    save_checkpoint: bool = True
    load_checkpoint_dir: Optional[str] = None
    
    # Optional evaluation configurations
    eval_interval: int = 500   # Run evaluation on validation split every N steps
    eval_steps: int = 50       # Number of validation steps to average over
    
    # Optional DeepSpeed configurations
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    
    # Hugging Face integration options
    hf_token: Optional[str] = None       # Write token for dataset and model upload access
    hf_repo_id: Optional[str] = None     # Hub repository identifier (e.g. org/repo_name)
    
    # Advanced acceleration options
    use_cuda_graphs: bool = False        # Capture and replay execution graph for faster steps

@dataclass
class PreprocessorDatasetConfig:
    """Config for a single dataset source in the preprocessing pipeline."""
    name: str  # Local path or HF dataset name
    source: str = "hf"  # "local" or "hf"
    config_name: Optional[str] = None
    split: str = "train"
    text_key: str = "text"
    target_tokens: int = -1  # Process everything by default

@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""
    datasets: List[PreprocessorDatasetConfig] = field(default_factory=list)
    output_path: str = "data/train.bin"
    tokenizer: str = "gpt2"
    vocab_limit: int = 32000
    hf_token: Optional[str] = None

@dataclass
class PicotronConfig:
    """Root configuration for Picotron."""
    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Helper to instantiate nested dataclasses from dictionaries without external libraries."""
    if not is_dataclass(cls):
        return data
    
    kwargs = {}
    cls_fields = {f.name: f for f in fields(cls)}
    for k, v in data.items():
        if k in cls_fields:
            field_type = cls_fields[k].type
            # Handle Optional wrapping
            if hasattr(field_type, "__origin__") and field_type.__origin__ is getattr(Optional, "__origin__", None):
                args = field_type.__args__
                non_none_args = [arg for arg in args if arg is not type(None)]
                if non_none_args:
                    field_type = non_none_args[0]

            # Handle List of dataclasses wrapping
            is_list = False
            list_item_type = None
            if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                is_list = True
                if field_type.__args__:
                    list_item_type = field_type.__args__[0]

            if is_dataclass(field_type) and isinstance(v, dict):
                kwargs[k] = _from_dict(field_type, v)
            elif is_list and list_item_type and is_dataclass(list_item_type) and isinstance(v, list):
                kwargs[k] = [_from_dict(list_item_type, item) for item in v]
            else:
                kwargs[k] = v
                
    # Fill defaults for missing fields
    from dataclasses import MISSING
    for name, f in cls_fields.items():
        if name not in kwargs:
            if f.default is not MISSING:
                kwargs[name] = f.default
            elif f.default_factory is not MISSING:
                kwargs[name] = f.default_factory()
                
    return cls(**kwargs)

def load_config_from_yaml(path: str) -> PicotronConfig:
    """Load configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _from_dict(PicotronConfig, data)

