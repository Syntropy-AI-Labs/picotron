"""
One-command data pipeline CLI: Clean, Tokenize, Shard, Pack, and Cache datasets.
"""

import os
import sys
import argparse
import re
import numpy as np
from datasets import load_dataset
from picotron.config import load_config_from_yaml
from picotron.data.async_tokenizer import AsyncTokenizerPipeline
from picotron.data.packing import pack_sequences

def clean_text(text: str) -> str:
    """Clean text by stripping HTML tags and normalizing whitespace."""
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def run_data_pipeline(config_path: str):
    print(f"Loading configuration from {config_path}...")
    cfg = load_config_from_yaml(config_path)
    prep_cfg = cfg.preprocess
    
    if not prep_cfg.datasets:
        print("No datasets configured.")
        return

    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(prep_cfg.output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Pipeline
    num_workers = prep_cfg.num_workers or 8
    pipeline = AsyncTokenizerPipeline(tokenizer_name=prep_cfg.tokenizer, num_workers=num_workers)
    
    all_raw_texts = []
    
    # 1. Download & Clean
    for idx, ds in enumerate(prep_cfg.datasets):
        print(f"\n--- Loading and Cleaning Dataset {idx+1}/{len(prep_cfg.datasets)}: {ds.name} ---")
        if ds.source == "local":
            if os.path.isfile(ds.name):
                with open(ds.name, "r", encoding="utf-8", errors="ignore") as f:
                    all_raw_texts.append(clean_text(f.read()))
        else:
            hf_tok = prep_cfg.hf_token if prep_cfg.hf_token is not None else cfg.train.hf_token
            dataset = load_dataset(ds.name, name=ds.config_name, split=ds.split, streaming=True, token=hf_tok)
            # Limit streaming to target tokens estimation (average 4 characters per token as a rough heuristic)
            char_limit = ds.target_tokens * 4 if ds.target_tokens > 0 else 100_000_000
            collected_chars = 0
            for row in dataset:
                txt = clean_text(row.get(ds.text_key, ""))
                if txt:
                    all_raw_texts.append(txt)
                    collected_chars += len(txt)
                if collected_chars >= char_limit:
                    break
                    
    # 2. Parallel Tokenization
    print(f"\nTokenizing {len(all_raw_texts):,} cleaned documents in parallel using {num_workers} processes...")
    tokenized_docs = pipeline.tokenize_parallel(all_raw_texts)
    
    # 3. Sequence Packing
    print(f"Packing tokenized documents into sequences of size {cfg.data.sequence_length}...")
    packed_sequences = pack_sequences(
        sequences=tokenized_docs,
        max_length=cfg.data.sequence_length,
        eos_token_id=2  # Default to GPT2/Standard EOS token ID
    )
    
    # 4. Sharding & Caching
    # Divide packed sequences into shards (e.g. 5 shards)
    num_shards = 5
    shard_size = len(packed_sequences) // num_shards
    
    print(f"Caching {len(packed_sequences):,} packed sequences into {num_shards} shards...")
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = len(packed_sequences) if shard_idx == num_shards - 1 else (shard_idx + 1) * shard_size
        
        shard_data = packed_sequences[start_idx:end_idx]
        # Flatten shard data
        flat_shard = [token for seq in shard_data for token in seq]
        
        # Write to shard file
        shard_path = f"{prep_cfg.output_path}.shard_{shard_idx}"
        with open(shard_path, "wb") as f_out:
            f_out.write(np.array(flat_shard, dtype=np.uint16).tobytes())
        print(f"Saved Shard {shard_idx} to: {shard_path} ({len(flat_shard):,} tokens)")
        
    # Write default complete cache
    flat_all = [token for seq in packed_sequences for token in seq]
    with open(prep_cfg.output_path, "wb") as f_out:
        f_out.write(np.array(flat_all, dtype=np.uint16).tobytes())
    print(f"\nCompleted! Unified packed cache saved to: {prep_cfg.output_path} ({len(flat_all):,} tokens)")

def main():
    parser = argparse.ArgumentParser(description="Picotron One-Command Data Pipeline")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()
    run_data_pipeline(args.config)

if __name__ == "__main__":
    main()
