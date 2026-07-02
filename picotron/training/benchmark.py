"""
Picotron performance benchmarking tool.
Measures throughput, GPU memory efficiency, GPU occupancy, and estimates TFLOPS / Model FLOPs Utilization (MFU).
"""

import sys
import os
import time
import argparse
import torch
import numpy as np

from picotron.config import load_config_from_yaml
from picotron.models.llama import LLaMAModel

def estimate_model_flops(model: torch.nn.Module, seq_len: int, batch_size: int) -> float:
    """
    Estimate floating-point operations (FLOPs) per forward + backward pass.
    Standard heuristic: 6 * P * tokens per step (where P is parameter count).
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokens = seq_len * batch_size
    return 6.0 * params * tokens

def draw_ascii_bar(val: float, max_val: float, width: int = 30) -> str:
    """Draws a clean console ascii progress bar."""
    filled = int(round(width * (val / max_val)))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {val:.1f}%"

def run_benchmark(config_path: str, num_steps: int = 50, warmup_steps: int = 10):
    print(f"Loading configuration from {config_path}...")
    cfg = load_config_from_yaml(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using benchmarking device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "a100" in gpu_name:
            peak_tflops = 312.0
        elif "h100" in gpu_name:
            peak_tflops = 1979.0
        elif "t4" in gpu_name:
            peak_tflops = 65.0
        elif "4090" in gpu_name:
            peak_tflops = 330.0
        else:
            peak_tflops = 150.0  # general assumption
    else:
        peak_tflops = 10.0

    print("Building model structure...")
    model = LLaMAModel(cfg.model).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable model parameters: {param_count:,} (~{param_count / 1e6:.1f}M)")

    # Prep inputs
    bsz = cfg.data.micro_batch_size
    seq_len = cfg.data.sequence_length
    print(f"Batch settings: Micro Batch Size = {bsz}, Sequence Length = {seq_len}")
    
    x = torch.randint(0, cfg.model.vocab_size, (bsz, seq_len), device=device)
    y = torch.randint(0, cfg.model.vocab_size, (bsz, seq_len), device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Warmup
    print(f"\nRunning {warmup_steps} warmup steps to compile structures...")
    for _ in range(warmup_steps):
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
    print(f"Benchmarking {num_steps} iterations...")
    step_times = []
    
    for step in range(num_steps):
        t0 = time.perf_counter()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    # Compile Stats
    avg_step_time = np.mean(step_times)
    tokens_per_step = bsz * seq_len
    tokens_per_sec = tokens_per_step / avg_step_time
    
    flops_per_step = estimate_model_flops(model, seq_len, bsz)
    achieved_tflops = (flops_per_step / avg_step_time) / 1e12
    mfu = (achieved_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0.0
    
    # Memory metrics
    if device == "cuda":
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)  # GB
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        mem_efficiency = (max_allocated / total_mem) * 100.0
    else:
        max_allocated = 0.0
        max_reserved = 0.0
        total_mem = 1.0
        mem_efficiency = 0.0

    print("\n" + "=" * 50)
    print("           PICOTRON PERFORMANCE BENCHMARK")
    print("=" * 50)
    print(f"Throughput:         {tokens_per_sec:.2f} tokens/sec")
    print(f"Average Step Time:  {avg_step_time * 1000:.2f} ms")
    print(f"Achieved Flops:     {achieved_tflops:.2f} TFLOPS")
    print(f"Peak GPU Capacity:  {peak_tflops:.2f} TFLOPS")
    print(f"Model FLOPs Util:   {mfu:.2f}%")
    if device == "cuda":
        print(f"Max Allocated Mem:  {max_allocated:.2f} GB / {total_mem:.2f} GB")
        print(f"Max Reserved Mem:   {max_reserved:.2f} GB")
    print("-" * 50)
    print("           HARDWARE EFFICIENCY GRAPHS")
    print("-" * 50)
    print(f"Model FLOPs Util (MFU):  {draw_ascii_bar(mfu, 100.0)}")
    print(f"GPU Memory Efficiency:   {draw_ascii_bar(mem_efficiency, 100.0)}")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Picotron Performance Benchmark Tool")
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument("--steps", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    args = parser.parse_args()
    
    run_benchmark(args.config, num_steps=args.steps, warmup_steps=args.warmup)

if __name__ == "__main__":
    main()
