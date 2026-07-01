"""
PEFT Model wrappers and utility methods to inject, save, and load adapters.
Compatible with Hugging Face PEFT style structures.
"""

import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from picotron.peft.lora import LoRALinear

class PeftModel(nn.Module):
    """
    Wrapper mapping low-rank adapters over selected modules of LLaMAModel.
    """
    def __init__(self, model: nn.Module, target_modules: List[str], r: int = 8, lora_alpha: int = 16, use_dora: bool = False):
        super().__init__()
        self.base_model = model
        self.target_modules = target_modules
        self.r = r
        self.lora_alpha = lora_alpha
        self.use_dora = use_dora
        
        self.peft_layers = {}
        self._inject_adapters()

    def _inject_adapters(self):
        """Scan the model tree and replace targeted linear layers with LoRA/DoRA layers."""
        # First freeze all parameters in base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        for name, module in self.base_model.named_modules():
            # Check if module matches name targets (e.g. q_proj, v_proj)
            is_target = any(target in name for target in self.target_modules)
            if is_target and isinstance(module, nn.Linear):
                # Retrieve parent module and attribute name
                parent = self._get_parent_module(name)
                attr_name = name.split(".")[-1]
                
                # Wrap with LoRALinear
                lora_layer = LoRALinear(
                    base_layer=module,
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    use_dora=self.use_dora
                )
                setattr(parent, attr_name, lora_layer)
                self.peft_layers[name] = lora_layer
                
        print(f"Successfully injected {len(self.peft_layers)} LoRA adapters into model.")

    def _get_parent_module(self, path: str) -> nn.Module:
        """Helper to navigate module path and get parent."""
        parts = path.split(".")
        curr = self.base_model
        for part in parts[:-1]:
            curr = getattr(curr, part)
        return curr

    def merge_adapters(self):
        """Merge all adapter weights into base layers for zero-latency inference."""
        for layer in self.peft_layers.values():
            layer.merge()

    def unmerge_adapters(self):
        """De-merge all adapter weights back to standard states."""
        for layer in self.peft_layers.values():
            layer.unmerge()

    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extract only the adapter parameters (excludes base model parameters)."""
        state_dict = {}
        for name, param in self.named_parameters():
            if "lora_" in name or ".m" in name:
                state_dict[name] = param.data
        return state_dict

    def save_pretrained(self, save_dir: str):
        """Save adapter weights and config metadata (compatible with HF)."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save adapter weights
        adapter_state = self.get_adapter_state_dict()
        torch.save(adapter_state, os.path.join(save_dir, "adapter_model.bin"))
        
        # Save config metadata
        config = {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "use_dora": self.use_dora,
            "peft_type": "LORA"
        }
        with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=4)
        print(f"Adapter saved successfully to: {save_dir}")

    def load_adapter(self, load_dir: str):
        """Load adapter weights back into injected layer parameters."""
        weights_path = os.path.join(load_dir, "adapter_model.bin")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No adapter weights found at {weights_path}")
            
        state_dict = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)
        print(f"Adapter weights loaded successfully from: {load_dir}")

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

def get_peft_model(model: nn.Module, target_modules: List[str], r: int = 8, lora_alpha: int = 16, use_dora: bool = False) -> PeftModel:
    """Wrapper initializer for PeftModel adapter injections."""
    return PeftModel(model, target_modules, r, lora_alpha, use_dora)
