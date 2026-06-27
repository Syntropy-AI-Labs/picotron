"""
Main Trainer loop for Picotron.
Orchestrates training cycles, gradient accumulation, AMP autocast, logging, and checkpointing.
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import save_file, load_file
from typing import Optional

from picotron.config import PicotronConfig
from picotron.utils.dtype import get_default_dtype
from picotron.utils.logging import logger
from picotron.optim.zero import ZeroRedundancyOptimizer

class Trainer:
    """
    Main trainer class for model pre-training.
    """
    def __init__(self, config: PicotronConfig, model: nn.Module, train_dataloader):
        """Initialize trainer state, optimizer, learning rate schedule, and devices."""
        self.config = config
        self.raw_model = model
        self.dataloader = train_dataloader
        
        # Setup distributed ranks
        self.distributed = dist.is_available() and dist.is_initialized()
        if self.distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            
        logger.set_rank(self.rank)
        
        # Setup Device and precision dtype
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.raw_model.to(self.device)
        
        # Setup mixed precision auto-detection
        if config.train.mixed_precision == "auto":
            self.amp_dtype = get_default_dtype()
        elif config.train.mixed_precision == "fp16":
            self.amp_dtype = torch.float16
        elif config.train.mixed_precision == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float32
            
        self.use_amp = self.amp_dtype in (torch.float16, torch.bfloat16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))
        
        # Wrap model with DDP if distributed
        if self.distributed:
            self.model = DDP(self.raw_model, device_ids=[local_rank])
        else:
            self.model = self.raw_model

        # JIT compile model if configured
        if config.train.compile:
            logger.info("Compiling model via torch.compile...")
            self.model = torch.compile(self.model)

        # Setup Optimizer (ZeRO-1 vs Standard AdamW)
        if self.distributed and config.parallel.zero_stage == 1:
            logger.info("Initializing ZeRO-1 optimizer...")
            self.optimizer = ZeroRedundancyOptimizer(
                self.raw_model.parameters(),
                torch.optim.AdamW,
                rank=self.rank,
                world_size=self.world_size,
                lr=config.train.learning_rate,
                betas=(config.train.adam_beta1, config.train.adam_beta2),
                eps=config.train.adam_eps,
                weight_decay=config.train.weight_decay
            )
        else:
            logger.info("Initializing standard AdamW optimizer...")
            self.optimizer = torch.optim.AdamW(
                self.raw_model.parameters(),
                lr=config.train.learning_rate,
                betas=(config.train.adam_beta1, config.train.adam_beta2),
                eps=config.train.adam_eps,
                weight_decay=config.train.weight_decay
            )
            
        self.global_step = 0
        if config.train.load_checkpoint_dir is not None:
            self.load_checkpoint(config.train.load_checkpoint_dir)

    def get_lr(self, step: int) -> float:
        """Calculate learning rate with linear warmup and cosine decay."""
        tc = self.config.train
        if step < tc.warmup_steps:
            return tc.learning_rate * (step + 1) / tc.warmup_steps
        
        decay_steps = tc.lr_decay_steps if tc.lr_decay_steps is not None else tc.max_steps
        if step > decay_steps:
            return tc.min_learning_rate
            
        progress = (step - tc.warmup_steps) / (decay_steps - tc.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return tc.min_learning_rate + cosine_decay * (tc.learning_rate - tc.min_learning_rate)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Run single step forward, backward, accumulation, and gradient update."""
        loss_accum = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        
        # Define cross entropy loss function
        criterion = nn.CrossEntropyLoss()
        
        # Manual gradient accumulation
        accum_steps = self.config.train.grad_accum_steps
        for micro_step in range(accum_steps):
            # Fetch slices for micro-batch
            mb_x = x[micro_step * x.size(0) // accum_steps : (micro_step + 1) * x.size(0) // accum_steps]
            mb_y = y[micro_step * y.size(0) // accum_steps : (micro_step + 1) * y.size(0) // accum_steps]
            
            mb_x = mb_x.to(self.device)
            mb_y = mb_y.to(self.device)
            
            # Context manager for mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.model(mb_x)
                # Reshape for cross entropy
                loss = criterion(logits.view(-1, logits.size(-1)), mb_y.view(-1))
                loss = loss / accum_steps
                
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            loss_accum += loss.item()

        # Update learning rate
        lr = self.get_lr(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Unscale gradients for clipping
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
            
        # Apply gradient clipping
        if self.config.train.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.config.train.grad_clip)
            
        # Step optimizer
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        self.global_step += 1
        return loss_accum

    def train(self) -> None:
        """Execute the primary training loop across configured maximum steps."""
        self.model.train()
        
        data_iter = iter(self.dataloader)
        
        # Initialize file-based metrics logger on rank 0
        metrics_file = None
        if self.rank == 0:
            os.makedirs(self.config.train.checkpoint_dir, exist_ok=True)
            metrics_path = os.path.join(self.config.train.checkpoint_dir, "metrics.csv")
            write_header = not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0
            metrics_file = open(metrics_path, "a", encoding="utf-8")
            if write_header:
                metrics_file.write("step,loss,lr\n")
                metrics_file.flush()
        
        # Use tqdm progress bar on rank 0
        from tqdm import tqdm
        pbar = tqdm(
            total=self.config.train.max_steps,
            initial=self.global_step,
            desc="Training Picotron",
            disable=(self.rank != 0)
        )
        
        while self.global_step < self.config.train.max_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                x, y = next(data_iter)
                
            loss = self.train_step(x, y)
            
            pbar.update(1)
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.2e}"})
            
            # Write metrics to log file
            if self.rank == 0 and metrics_file is not None:
                metrics_file.write(f"{self.global_step},{loss:.6f},{lr:.6e}\n")
                if self.global_step % 10 == 0:
                    metrics_file.flush()
                
            # Checkpoint save
            if self.global_step % self.config.train.checkpoint_interval == 0:
                self.save_checkpoint()
                
        pbar.close()
        if metrics_file is not None:
            metrics_file.close()

    def save_checkpoint(self) -> None:
        """Persist training weights and optimizer state to files."""
        if self.rank != 0 or not self.config.train.save_checkpoint:
            return
            
        out_dir = os.path.join(self.config.train.checkpoint_dir, f"step_{self.global_step}")
        os.makedirs(out_dir, exist_ok=True)
        
        # Save model using safetensors
        state_dict = self.raw_model.state_dict()
        model_path = os.path.join(out_dir, "model.safetensors")
        save_file(state_dict, model_path)
        
        # Save config and scheduler stats
        torch.save({
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict()
        }, os.path.join(out_dir, "training_state.pt"))
        logger.info(f"Checkpoint saved at: {out_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Restore weights and states from checkpoint folder."""
        # Find paths
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        
        if os.path.exists(model_path):
            state_dict = load_file(model_path)
            self.raw_model.load_state_dict(state_dict)
            
        if os.path.exists(state_path):
            checkpoint = torch.load(state_path, map_location=self.device)
            self.global_step = checkpoint["global_step"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Loaded checkpoint from: {checkpoint_path} (step: {self.global_step})")
