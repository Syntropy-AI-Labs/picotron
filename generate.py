"""
Inference generation script for Picotron.
Loads a saved model checkpoint and generates text autoregressively.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from picotron.config import load_config_from_yaml
from picotron.models.llama import LLaMAModel

@torch.no_grad()
def generate(
    model: LLaMAModel, 
    prompt: str, 
    tokenizer: AutoTokenizer, 
    max_new_tokens: int = 50, 
    temperature: float = 0.8
) -> str:
    """Generate tokens autoregressively from a starting prompt."""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize input and move to device
    # Modulo-clamping to match model's vocab_size limit (32000)
    input_ids = torch.tensor([[t % model.vocab_size for t in tokenizer.encode(prompt)]], device=device)
    
    for _ in range(max_new_tokens):
        # Crop context to model's maximum sequence length
        cond_ids = input_ids[:, -model.config.max_position_embeddings:]
        
        # Forward pass to get logits for the last token
        logits = model(cond_ids)[:, -1, :]
        
        # Apply temperature scaling
        if temperature > 0.0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
        # Append generated token to sequence
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
    # Decode back to text
    return tokenizer.decode(input_ids[0].tolist())

def main():
    config_path = "E:/AI/2M.yaml" # Config path
    checkpoint_dir = "checkpoints/step_1000" # Path to your saved checkpoint
    
    # 1. Load config
    cfg = load_config_from_yaml(config_path)
    
    # 2. Build model structure
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LLaMAModel structure to {device}...")
    model = LLaMAModel(cfg.model)
    
    # 3. Load weights from safetensors checkpoint
    from safetensors.torch import load_file
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: No weights found at {weights_path}. Running with random initialization.")
        
    model.to(device)
    
    # 4. Initialize tokenizer (matches the training dataset tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 5. Generate
    prompt = "Once upon a time in a school"
    print(f"\nPrompt: {prompt}")
    output = generate(model, prompt, tokenizer, max_new_tokens=50, temperature=0.7)
    print(f"\nGenerated output:\n{output}")

if __name__ == "__main__":
    main()
