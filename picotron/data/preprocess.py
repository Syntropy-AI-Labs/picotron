"""
CLI utility for processing, tokenizing, and preparing datasets for Picotron.
Sequentially tokenizes multiple local files or HF datasets based on a YAML config.
"""

import os
import sys
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from picotron.config import load_config_from_yaml

# Suppress Hugging Face warnings in tokenizers
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def preprocess_local(input_path: str, tokenizer, f_out, vocab_limit: int, target_tokens: int):
    """Tokenize a local file or directory of text files and append to the output stream."""
    files_to_process = []
    if os.path.isfile(input_path):
        files_to_process.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".txt"):
                    files_to_process.append(os.path.join(root, file))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not files_to_process:
        print(f"No text files (.txt) found in: {input_path}")
        return 0

    print(f"Tokenizing local files from: {input_path}")
    tokens_saved = 0
    token_accumulator = []
    
    pbar = tqdm(total=target_tokens if target_tokens > 0 else None, unit="tokens", desc="Processing Local")

    for file_path in files_to_process:
        if target_tokens > 0 and tokens_saved >= target_tokens:
            break
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as tf:
                text = tf.read()
                
            enc = [t % vocab_limit for t in tokenizer.encode(text)]
            token_accumulator.extend(enc)
            
            if len(token_accumulator) >= 2_000_000:
                chunk_size = 2_000_000
                if target_tokens > 0 and tokens_saved + chunk_size > target_tokens:
                    chunk_size = target_tokens - tokens_saved
                
                chunk = np.array(token_accumulator[:chunk_size], dtype=np.uint16)
                chunk.tofile(f_out)
                tokens_saved += chunk_size
                pbar.update(chunk_size)
                token_accumulator = token_accumulator[chunk_size:]
        except Exception as e:
            print(f"\nError processing file {file_path}: {e}")

    # Flush remaining
    if token_accumulator and (target_tokens <= 0 or tokens_saved < target_tokens):
        chunk_size = len(token_accumulator)
        if target_tokens > 0 and tokens_saved + chunk_size > target_tokens:
            chunk_size = target_tokens - tokens_saved
        chunk = np.array(token_accumulator[:chunk_size], dtype=np.uint16)
        chunk.tofile(f_out)
        tokens_saved += chunk_size
        pbar.update(chunk_size)
        
    pbar.close()
    return tokens_saved


def preprocess_hf(dataset_name: str, config_name: str, split: str, text_key: str, tokenizer, f_out, vocab_limit: int, target_tokens: int, hf_token: str):
    """Stream and tokenize a dataset from Hugging Face Hub and append to output stream."""
    from datasets import load_dataset

    print(f"Streaming dataset '{dataset_name}' (config: '{config_name or 'default'}', split: '{split}')...")
    dataset = load_dataset(
        dataset_name,
        name=config_name,
        split=split,
        streaming=True,
        token=hf_token
    )

    tokens_saved = 0
    token_accumulator = []
    
    pbar = tqdm(total=target_tokens if target_tokens > 0 else None, unit="tokens", desc=f"Processing {dataset_name.split('/')[-1]}")

    for row in dataset:
        if target_tokens > 0 and tokens_saved >= target_tokens:
            break
            
        text = row.get(text_key, "")
        if not text:
            continue

        enc = [t % vocab_limit for t in tokenizer.encode(text)]
        token_accumulator.extend(enc)

        if len(token_accumulator) >= 2_000_000:
            chunk_size = 2_000_000
            if target_tokens > 0 and tokens_saved + chunk_size > target_tokens:
                chunk_size = target_tokens - tokens_saved
            
            chunk = np.array(token_accumulator[:chunk_size], dtype=np.uint16)
            chunk.tofile(f_out)
            tokens_saved += chunk_size
            pbar.update(chunk_size)
            token_accumulator = token_accumulator[chunk_size:]

    # Flush remaining
    if token_accumulator and (target_tokens <= 0 or tokens_saved < target_tokens):
        chunk_size = len(token_accumulator)
        if target_tokens > 0 and tokens_saved + chunk_size > target_tokens:
            chunk_size = target_tokens - tokens_saved
        chunk = np.array(token_accumulator[:chunk_size], dtype=np.uint16)
        chunk.tofile(f_out)
        tokens_saved += chunk_size
        pbar.update(chunk_size)

    pbar.close()
    return tokens_saved


def main():
    if len(sys.argv) < 2:
        print("Usage: picotron-preprocess <path_to_config.yaml>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    cfg = load_config_from_yaml(config_path)
    prep_cfg = cfg.preprocess
    
    if not prep_cfg.datasets:
        print("No datasets configured in preprocess.datasets.")
        return

    # Delete previous binary file if it exists to start fresh
    if os.path.exists(prep_cfg.output_path):
        os.remove(prep_cfg.output_path)
        print(f"Cleared existing output file: {prep_cfg.output_path}")
        
    os.makedirs(os.path.dirname(os.path.abspath(prep_cfg.output_path)), exist_ok=True)
    
    print(f"Loading tokenizer '{prep_cfg.tokenizer}'...")
    tokenizer = AutoTokenizer.from_pretrained(prep_cfg.tokenizer)
    
    total_tokens_written = 0
    
    # Open binary file in append mode to process multiple sources
    with open(prep_cfg.output_path, "wb") as f_out:
        for idx, ds in enumerate(prep_cfg.datasets):
            print(f"\n--- Dataset Source {idx+1}/{len(prep_cfg.datasets)}: {ds.name} ---")
            
            if ds.source == "local":
                tokens = preprocess_local(
                    input_path=ds.name,
                    tokenizer=tokenizer,
                    f_out=f_out,
                    vocab_limit=prep_cfg.vocab_limit,
                    target_tokens=ds.target_tokens
                )
            else:
                hf_tok = prep_cfg.hf_token if prep_cfg.hf_token is not None else cfg.train.hf_token
                tokens = preprocess_hf(
                    dataset_name=ds.name,
                    config_name=ds.config_name,
                    split=ds.split,
                    text_key=ds.text_key,
                    tokenizer=tokenizer,
                    f_out=f_out,
                    vocab_limit=prep_cfg.vocab_limit,
                    target_tokens=ds.target_tokens,
                    hf_token=hf_tok
                )
                
            total_tokens_written += tokens
            
    print(f"\nPreprocessing successfully finished! Total tokens compiled: {total_tokens_written:,}")

if __name__ == "__main__":
    main()
