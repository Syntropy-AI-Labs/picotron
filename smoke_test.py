"""
Smoke test to verify Picotron model instantiations, forward passes, and backward passes on CPU.
Does not require any GPU or optional GPU-specific dependencies.
"""

import torch
from picotron.config import ModelConfig
from picotron.models.llama import LLaMAModel

def run_smoke_test() -> None:
    """Run forward and backward pass on a tiny LLaMAModel instance."""
    print("Initializing tiny LLaMAModel configuration...")
    # Tiny model configurations
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )
    
    # Force cpu execution
    model = LLaMAModel(config)
    model.cpu()
    
    # Fake batch: size=2, sequence_length=32
    print("Generating mock inputs...")
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    
    print("Running model forward pass...")
    logits, aux_loss = model(input_ids)
    
    # Check shape: [batch, sequence_length, vocab_size]
    expected_shape = (2, 32, config.vocab_size)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    
    print("Running model backward pass...")
    # Target label tokens (shifted left)
    targets = torch.randint(0, config.vocab_size, (2, 32))
    
    # Cross entropy loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, config.vocab_size), targets.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Validate loss value
    loss_val = loss.item()
    assert torch.isfinite(loss), f"Loss is not finite: {loss_val}"
    
    print(f"Loss value: {loss_val:.4f}")
    print("PICOTRON SMOKE TEST PASSED")

if __name__ == "__main__":
    run_smoke_test()
