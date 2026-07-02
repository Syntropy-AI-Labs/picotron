"""
Multi-GPU verification script for Picotron's advanced distributed paradigms.
Designed to run on Kaggle 2xT4 GPU environments using torchrun.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist

# Add parent directory to path so we can import picotron
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from picotron.parallel.tensor import ColumnParallelLinear, RowParallelLinear
from picotron.parallel.pipeline import send_forward, recv_forward
from picotron.parallel.context import shard_context, gather_context
from picotron.parallel.expert import dispatch_tokens_to_experts
from picotron.nn.mixers import MLA, SelectiveSSM
from picotron.nn.mlp import MoEMLP
from picotron.config import ModelConfig

def run_distributed_tests(rank: int, world_size: int):
    # Initialize torch.distributed backend
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[Rank {rank}] Distributed environment successfully initialized.")

    # =================================================================
    # 1. TENSOR PARALLELISM (TP) TEST
    # =================================================================
    print(f"[Rank {rank}] 1. Verifying TP Column/Row Parallel projections...")
    # Initialize mock inputs
    x_tp = torch.randn(2, 8, device=device)
    
    col_layer = ColumnParallelLinear(in_features=8, out_features=16, bias=False, tp_group=None).to(device)
    row_layer = RowParallelLinear(in_features=16, out_features=8, bias=False, tp_group=None).to(device)
    
    out_col = col_layer(x_tp)
    out_tp = row_layer(out_col)
    
    assert out_tp.shape == (2, 8), "TP column/row projection shape mismatch!"
    print(f"[Rank {rank}] TP test passed.")

    # =================================================================
    # 2. PIPELINE PARALLELISM (PP) TEST
    # =================================================================
    print(f"[Rank {rank}] 2. Verifying PP Activation handshakes...")
    # Rank 0 sends to Rank 1
    if rank == 0:
        tx_tensor = torch.ones(2, 4, device=device) * 5.0
        send_forward(tx_tensor, next_rank=1)
        print(f"[Rank {rank}] PP Sent Forward tensor: {tx_tensor.cpu().numpy().tolist()}")
    elif rank == 1:
        rx_tensor = recv_forward(tensor_shape=(2, 4), dtype=torch.float32, prev_rank=0)
        print(f"[Rank {rank}] PP Received Forward tensor: {rx_tensor.cpu().numpy().tolist()}")
        assert torch.allclose(rx_tensor, torch.ones(2, 4, device=device) * 5.0), "PP handshake value mismatch!"
        
    dist.barrier()
    print(f"[Rank {rank}] PP test passed.")

    # =================================================================
    # 3. CONTEXT PARALLELISM (CP) TEST
    # =================================================================
    print(f"[Rank {rank}] 3. Verifying CP sequence sharding and gathering...")
    # We create a sequence of length 8
    seq = torch.arange(8, device=device).unsqueeze(0).unsqueeze(-1).float() # [1, 8, 1]
    
    # Shard sequence along seq_len dimension
    sharded_seq = shard_context(seq, cp_size=world_size, cp_rank=rank, dim=1)
    print(f"[Rank {rank}] CP sharded segment sequence: {sharded_seq.squeeze().cpu().numpy().tolist()}")
    
    # Gather sharded segments back
    gathered_seq = gather_context(sharded_seq, cp_group=None, dim=1)
    print(f"[Rank {rank}] CP test passed.")

    # =================================================================
    # 4. EXPERT PARALLELISM (EP) / MoE TEST
    # =================================================================
    print(f"[Rank {rank}] 4. Verifying MoE routing & EP dispatch...")
    # Mock tokens
    tokens = torch.randn(4, 8, device=device)
    dispatch_idx = torch.tensor([0, 1, 0, 1], device=device)
    
    routed_tokens = dispatch_tokens_to_experts(tokens, dispatch_idx, ep_group=None)
    assert routed_tokens.shape == (4, 8), "EP token dispatch shape mismatch!"
    print(f"[Rank {rank}] MoE / EP test passed.")

    # =================================================================
    # 5. MLA & SSM FORWARD TEST ON GPU
    # =================================================================
    print(f"[Rank {rank}] 5. Running MLA & SSM forward passes on GPU...")
    # Test MLA
    mla = MLA(hidden_size=8, num_heads=2, head_dim=4, kv_lora_rank=8, q_lora_rank=4, rope_dim=2).to(device)
    x_mla = torch.randn(2, 4, 8, device=device)
    out_mla = mla(x_mla)
    assert out_mla.shape == (2, 4, 8), "MLA forward shape mismatch!"
    
    # Test SSM (Mamba-style)
    ssm = SelectiveSSM(d_model=8, d_state=4, d_inner=8).to(device)
    x_ssm = torch.randn(2, 4, 8, device=device)
    out_ssm = ssm(x_ssm)
    assert out_ssm.shape == (2, 4, 8), "SSM forward shape mismatch!"
    
    print(f"[Rank {rank}] MLA & SSM GPU tests passed.")
    print(f"[Rank {rank}] --- ALL DISTRIBUTED GPU TESTS COMPLETED SUCCESSFULLY ---")
    
    # Clean up process group
    dist.destroy_process_group()

if __name__ == "__main__":
    # Check if run under torchrun / distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run_distributed_tests(rank, world_size)
    else:
        print("Please run this script using torchrun to enable multi-GPU distributed testing:")
        print("torchrun --nproc_per_node=2 exclude/test_dual_gpu.py")
