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
    
    # Clean previous output
    if os.path.exists(prep_cfg.output_path):
        os.remove(prep_cfg.output_path)
    
    # Initialize Pipeline
    num_workers = prep_cfg.num_workers or 8
    pipeline = AsyncTokenizerPipeline(tokenizer_name=prep_cfg.tokenizer, num_workers=num_workers)
    
    total_tokens_written = 0
    chunk_raw_texts = []
    chunk_size_docs = 2000
    
    def process_and_write_chunk():
        nonlocal total_tokens_written, chunk_raw_texts
        if not chunk_raw_texts:
            return
        
        # Parallel Tokenization
        tokenized_docs = pipeline.tokenize_parallel(chunk_raw_texts)
        
        # Sequence Packing
        packed_sequences = pack_sequences(
            sequences=tokenized_docs,
            max_length=cfg.data.sequence_length,
            eos_token_id=2  # Standard GPT-2 EOS token
        )
        
        if packed_sequences:
            flat_tokens = [token for seq in packed_sequences for token in seq]
            # Write directly to disk
            with open(prep_cfg.output_path, "ab") as f_out:
                f_out.write(np.array(flat_tokens, dtype=np.uint16).tobytes())
            total_tokens_written += len(flat_tokens)
            print(f"Tokenized and packed {len(chunk_raw_texts):,} docs. Total tokens saved to disk: {total_tokens_written:,}")
        
        # Clear memory
        chunk_raw_texts = []

    # 1. Download & Clean & Process on the fly
    for idx, ds in enumerate(prep_cfg.datasets):
        print(f"\n--- Loading and Processing Dataset {idx+1}/{len(prep_cfg.datasets)}: {ds.name} ---")
        if ds.source == "local":
            if os.path.isfile(ds.name):
                with open(ds.name, "r", encoding="utf-8", errors="ignore") as f:
                    chunk_raw_texts.append(clean_text(f.read()))
                if len(chunk_raw_texts) >= chunk_size_docs:
                    process_and_write_chunk()
        else:
            hf_tok = prep_cfg.hf_token if prep_cfg.hf_token is not None else cfg.train.hf_token
            dataset = load_dataset(ds.name, name=ds.config_name, split=ds.split, streaming=True, token=hf_tok)
            
            char_limit = ds.target_tokens * 4 if ds.target_tokens > 0 else 100_000_000
            collected_chars = 0
            
            for row in dataset:
                txt = clean_text(row.get(ds.text_key, ""))
                if txt:
                    chunk_raw_texts.append(txt)
                    collected_chars += len(txt)
                
                if len(chunk_raw_texts) >= chunk_size_docs:
                    process_and_write_chunk()
                    
                if collected_chars >= char_limit:
                    break
                    
    # Process final remaining chunk
    if chunk_raw_texts:
        process_and_write_chunk()
        
    print(f"\nCompleted! Unified packed cache saved to: {prep_cfg.output_path} (Total: {total_tokens_written:,} tokens)")

def main():
    parser = argparse.ArgumentParser(description="Picotron One-Command Data Pipeline")
    parser.add_argument("config", help="Path to config.yaml")
    args = parser.parse_args()
    run_data_pipeline(args.config)

if __name__ == "__main__":
    main()
