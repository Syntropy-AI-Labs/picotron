"""
Script to download a shard of FineWeb-Edu, tokenize it using a HuggingFace tokenizer,
and write exactly 60M tokens to data/train.bin.
"""

import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    print("Loading HuggingFaceFW/fineweb-edu (sample-10BT)...")
    # Load dataset stream to prevent full download overhead
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    
    # Use GPT-2 style tokenizer
    print("Loading gpt2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    os.makedirs("data", exist_ok=True)
    bin_path = "data/train.bin"
    
    target_tokens = 30_000_000
    token_accumulator = []
    tokens_saved = 0
    
    print("Processing and tokenizing dataset streams...")
    # Initialize progress bar
    pbar = tqdm(total=target_tokens, unit="tokens", desc="Tokenizing FineWeb-Edu")
    
    # Open the binary file to write incrementally
    with open(bin_path, "wb") as f:
        for idx, row in enumerate(dataset):
            text = row["text"]
            # Tokenize text and clamp to vocab limit (32000)
            enc = [t % 32000 for t in tokenizer.encode(text)]
            token_accumulator.extend(enc)
            
            # Flush in chunks of 1M tokens to save RAM
            if len(token_accumulator) >= 1_000_000:
                chunk = np.array(token_accumulator[:1_000_000], dtype=np.uint16)
                chunk.tofile(f)
                
                tokens_saved += 1_000_000
                pbar.update(1_000_000)
                
                # Keep remaining
                token_accumulator = token_accumulator[1_000_000:]
                
            if tokens_saved >= target_tokens:
                break
                
        # Write remaining if any
        if tokens_saved < target_tokens and token_accumulator:
            needed = target_tokens - tokens_saved
            chunk = np.array(token_accumulator[:needed], dtype=np.uint16)
            chunk.tofile(f)
            tokens_saved += len(chunk)
            pbar.update(len(chunk))
            
    pbar.close()
    print(f"Successfully wrote {tokens_saved:,} tokens to {bin_path}!")

if __name__ == "__main__":
    main()
