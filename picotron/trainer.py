"""
Main Trainer loop for Picotron.
Orchestrates training cycles, validation evaluations, gradient accumulation, AMP autocast, logging, and checkpointing.
Includes optimized PyTorch accelerations: prefetching dataloaders, async background checkpoints, memory pooling, and CUDA Graphs.
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import save_file, load_file
import threading
from typing import Optional

from picotron.config import PicotronConfig
from picotron.utils.dtype import get_default_dtype
from picotron.utils.logging import logger
from picotron.optim.zero import ZeroRedundancyOptimizer

class PrefetchDataloader:
    """
    Asynchronously prefetches the next batch of data to device memory.
    Overlaps CPU-to-GPU memory transfer with GPU computing.
    """
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.iter = None
        self.next_x = None
        self.next_y = None
        
    def __len__(self):
        return len(self.dataloader)
        
    def __iter__(self):
        self.iter = iter(self.dataloader)
        self.preload()
        return self
        
    def preload(self):
        try:
            self.next_x, self.next_y = next(self.iter)
        except StopIteration:
            self.next_x, self.next_y = None, None
            return
            
        with torch.cuda.stream(self.stream):
            self.next_x = self.next_x.to(self.device, non_blocking=True)
            self.next_y = self.next_y.to(self.device, non_blocking=True)
            
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        x = self.next_x
        y = self.next_y
        
        if x is None:
            raise StopIteration
            
        self.preload()
        return x, y

class Trainer:
    """
    Main trainer class for model pre-training.
    """
    def __init__(self, config: PicotronConfig, model: nn.Module, train_dataloader, val_dataloader=None):
        """Initialize trainer state, optimizer, learning rate schedule, and devices."""
        self.config = config
        self.raw_model = model
        
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
        
        # Configure Caching Allocator Memory Pool to prevent allocations overhead
        if self.device.type == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
            torch.cuda.set_per_process_memory_fraction(0.95, device=self.device)
            logger.info("Memory Pooling caching allocations enabled (95% GPU capacity limit).")
        
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
        
        self.use_deepspeed = config.train.use_deepspeed
        if self.use_deepspeed:
            logger.info("Initializing DeepSpeed Engine...")
            try:
                import deepspeed
            except ImportError:
                raise ImportError("deepspeed is not installed. Please install it using `pip install deepspeed` to use DeepSpeed features.")
                
            ds_config = config.train.deepspeed_config
            if ds_config is None:
                # Basic default config supporting ZeRO-2
                ds_config = {
                    "train_micro_batch_size_per_gpu": config.data.micro_batch_size,
                    "gradient_accumulation_steps": config.train.grad_accum_steps,
                    "zero_optimization": {
                        "stage": 2,
                        "allgather_partitions": True,
                        "overlap_comm": True,
                        "reduce_scatter": True,
                        "contiguous_gradients": True
                    },
                    "fp16": {
                        "enabled": self.amp_dtype == torch.float16
                    },
                    "bf16": {
                        "enabled": self.amp_dtype == torch.bfloat16
                    },
                    "optimizer": {
                        "type": "AdamW",
                        "params": {
                            "lr": config.train.learning_rate,
                            "betas": [config.train.adam_beta1, config.train.adam_beta2],
                            "eps": config.train.adam_eps,
                            "weight_decay": config.train.weight_decay
                        }
                    }
                }
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model=self.raw_model,
                model_parameters=self.raw_model.parameters(),
                config=ds_config
            )
            self.dataloader = train_dataloader
            self.val_dataloader = val_dataloader
        else:
            # Wrap model with DDP if distributed
            if self.distributed:
                self.model = DDP(self.raw_model, device_ids=[local_rank])
            else:
                self.model = self.raw_model

            # Wrap dataloaders with asynchronous prefetcher if running on GPU
            if self.device.type == "cuda":
                logger.info("Enabling asynchronous CUDA prefetching dataloaders...")
                self.dataloader = PrefetchDataloader(train_dataloader, self.device)
                self.val_dataloader = PrefetchDataloader(val_dataloader, self.device) if val_dataloader is not None else None
            else:
                self.dataloader = train_dataloader
                self.val_dataloader = val_dataloader

            # Setup Optimizer (ZeRO-1 vs Fused Standard AdamW)
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
                logger.info("Initializing standard AdamW optimizer (Fused = True)...")
                # Leverage PyTorch 2.0+ optimized fused AdamW kernel
                use_fused = (self.device.type == "cuda")
                self.optimizer = torch.optim.AdamW(
                    self.raw_model.parameters(),
                    lr=config.train.learning_rate,
                    betas=(config.train.adam_beta1, config.train.adam_beta2),
                    eps=config.train.adam_eps,
                    weight_decay=config.train.weight_decay,
                    fused=use_fused
                )
            
        # CUDA Graphs Initialization
        self.use_cuda_graphs = config.train.use_cuda_graphs and self.device.type == "cuda" and not self.use_deepspeed
        self.cuda_graph = None
        self.static_x = None
        self.static_y = None
        self.static_loss = None
        
        self.global_step = 0
        if config.train.load_checkpoint_dir is not None:
            self.load_checkpoint(config.train.load_checkpoint_dir)

    def get_lr(self, step: int) -> float:
        """Calculate learning rate with linear warmup and cosine decay."""
        tc = self.config.train
        if step < tc.warmup_steps:
            return tc.learning_rate * step / max(1, tc.warmup_steps)
            
        decay_steps = tc.lr_decay_steps if tc.lr_decay_steps is not None else tc.max_steps
        if step > decay_steps:
            return tc.min_learning_rate
            
        progress = (step - tc.warmup_steps) / (decay_steps - tc.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return tc.min_learning_rate + cosine_decay * (tc.learning_rate - tc.min_learning_rate)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on the validation dataloader and return average loss."""
        if self.val_dataloader is None:
            return 0.0
            
        self.model.eval()
        val_iter = iter(self.val_dataloader)
        val_loss_accum = 0.0
        eval_steps = self.config.train.eval_steps
        
        for step in range(eval_steps):
            try:
                x, y = next(val_iter)
            except StopIteration:
                val_iter = iter(self.val_dataloader)
                x, y = next(val_iter)
                
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                logits, _ = self.model(x)
                from picotron.kernels.triton_kernels import triton_cross_entropy
                loss = triton_cross_entropy(logits, y)
                
            val_loss_accum += loss.item()
            
        avg_val_loss = val_loss_accum / eval_steps
        
        if self.distributed:
            tensor_loss = torch.tensor(avg_val_loss, device=self.device)
            dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
            avg_val_loss = tensor_loss.item() / self.world_size
            
        self.model.train()
        return avg_val_loss

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Run single step forward, backward, accumulation, and gradient update."""
        if self.use_deepspeed:
            mb_x = x.to(self.device)
            mb_y = y.to(self.device)
            
            logits, aux_loss = self.model(mb_x)
            loss_ce = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), mb_y.view(-1))
            loss = loss_ce + 0.01 * aux_loss
            
            self.model.backward(loss)
            self.model.step()
            
            loss_accum = loss_ce.item()
            
            lr = self.get_lr(self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            self.global_step += 1
            return loss_accum

        # -------------------------------------------------------------
        # CUDA Graphs Acceleration Path
        # -------------------------------------------------------------
        if self.use_cuda_graphs:
            # Initialize static input placeholders on first call
            if self.static_x is None:
                mb_shape_x = (x.size(0), x.size(1))
                mb_shape_y = (y.size(0), y.size(1))
                
                self.static_x = torch.empty(mb_shape_x, dtype=x.dtype, device=self.device)
                self.static_y = torch.empty(mb_shape_y, dtype=y.dtype, device=self.device)

            if self.cuda_graph is None:
                logger.info("Warming up caching allocator for CUDA Graphs...")
                # Run 3 warmup steps to stabilize memory allocations
                for _ in range(3):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.static_x.copy_(x[:self.static_x.size(0)])
                    self.static_y.copy_(y[:self.static_y.size(0)])
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        logits, aux_loss = self.model(self.static_x)
                        from picotron.kernels.triton_kernels import triton_cross_entropy
                        loss = triton_cross_entropy(logits, self.static_y) + 0.01 * aux_loss
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                logger.info("Capturing steps using CUDA Graph...")
                self.cuda_graph = torch.cuda.CUDAGraph()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Record static execution flow
                with torch.cuda.graph(self.cuda_graph):
                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        logits, aux_loss = self.model(self.static_x)
                        from picotron.kernels.triton_kernels import triton_cross_entropy
                        loss = triton_cross_entropy(logits, self.static_y) + 0.01 * aux_loss
                        
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                logger.info("CUDA Graph successfully captured!")

            # Copy batch inputs to placeholders
            self.static_x.copy_(x[:self.static_x.size(0)], non_blocking=True)
            self.static_y.copy_(y[:self.static_y.size(0)], non_blocking=True)
            
            # Replay captured forward/backward steps directly on the GPU
            self.cuda_graph.replay()
            
            # Step optimizer outside captured graph
            lr = self.get_lr(self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.global_step += 1
            return loss.item()

        # -------------------------------------------------------------
        # Standard Native Training Path
        # -------------------------------------------------------------
        loss_accum = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        
        accum_steps = self.config.train.grad_accum_steps
        for micro_step in range(accum_steps):
            mb_x = x[micro_step * x.size(0) // accum_steps : (micro_step + 1) * x.size(0) // accum_steps]
            mb_y = y[micro_step * y.size(0) // accum_steps : (micro_step + 1) * y.size(0) // accum_steps]
            
            mb_x = mb_x.to(self.device, non_blocking=True)
            mb_y = mb_y.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                logits, aux_loss = self.model(mb_x)
                
                # Fused Cross Entropy Triton Kernel
                from picotron.kernels.triton_kernels import triton_cross_entropy
                loss_ce = triton_cross_entropy(logits, mb_y)
                loss = (loss_ce + 0.01 * aux_loss) / accum_steps
                
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            loss_accum += loss_ce.item() / accum_steps

        # Update learning rate
        lr = self.get_lr(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
            
        if self.config.train.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.config.train.grad_clip)
            
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
                metrics_file.write("step,loss,lr,val_loss\n")
                metrics_file.flush()
        
        # Use tqdm progress bar on rank 0
        from tqdm import tqdm
        pbar = tqdm(
            total=self.config.train.max_steps,
            initial=self.global_step,
            desc="Training Picotron",
            disable=(self.rank != 0)
        )
        
        val_loss_str = ""
        
        while self.global_step < self.config.train.max_steps:
            # Trigger periodic evaluation if a validation dataloader is present
            if self.val_dataloader is not None and self.global_step % self.config.train.eval_interval == 0:
                val_loss = self.evaluate()
                val_loss_str = f"{val_loss:.4f}"
                logger.info(f"Evaluation at step {self.global_step}: Val Loss = {val_loss_str}")

            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                x, y = next(data_iter)
                
            loss = self.train_step(x, y)
            
            pbar.update(1)
            lr = self.optimizer.param_groups[0]['lr']
            
            pbar_metrics = {"loss": f"{loss:.4f}", "lr": f"{lr:.2e}"}
            if val_loss_str:
                pbar_metrics["val_loss"] = val_loss_str
            pbar.set_postfix(pbar_metrics)
            
            # Write metrics to log file
            if self.rank == 0 and metrics_file is not None:
                metrics_file.write(f"{self.global_step},{loss:.6f},{lr:.6e},{val_loss_str}\n")
                if self.global_step % 10 == 0:
                    metrics_file.flush()
                
            # Checkpoint save
            if self.global_step % self.config.train.checkpoint_interval == 0:
                self.save_checkpoint()
                
        pbar.close()
        # Save one final checkpoint at the end of training
        self.save_checkpoint()
        if metrics_file is not None:
            metrics_file.close()

    def save_checkpoint(self) -> None:
        """Persist training weights and optimizer state to files asynchronously."""
        if self.rank != 0 or not self.config.train.save_checkpoint:
            return
            
        step_dir = os.path.join(self.config.train.checkpoint_dir, f"step_{self.global_step}")
        os.makedirs(step_dir, exist_ok=True)
        
        weights_path = os.path.join(step_dir, "model.safetensors")
        state_path = os.path.join(step_dir, "training_state.pt")
        
        # Prepare state dict metadata
        state_dict = {
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
        }
        if self.scaler.is_enabled():
            state_dict["scaler_state"] = self.scaler.state_dict()

        # Copy raw model weights to CPU asynchronously to avoid blocking CUDA execution stream
        model_state_cpu = {k: v.cpu() for k, v in self.raw_model.state_dict().items()}
        
        # Run saving tasks in a background thread
        def async_save_task(step_dir, model_state, state_dict, weights_path, state_path, hf_repo_id, hf_token, global_step):
            try:
                save_file(model_state, weights_path)
                torch.save(state_dict, state_path)
                logger.info(f"Asynchronous checkpoint saved at: {step_dir}")
                
                # Optional Hugging Face Hub checkpoint upload
                if hf_repo_id is not None:
                    logger.info(f"Uploading checkpoint step_{global_step} to Hugging Face Hub: {hf_repo_id}...")
                    from huggingface_hub import HfApi, create_repo
                    create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True, token=hf_token)
                    api = HfApi(token=hf_token)
                    api.upload_folder(
                        folder_path=step_dir,
                        repo_id=hf_repo_id,
                        repo_type="model",
                        path_in_repo=f"checkpoint-step-{global_step}"
                    )
                    logger.info(f"Checkpoint uploaded successfully to HF Hub: {hf_repo_id}/checkpoint-step-{global_step}")
            except Exception as e:
                logger.error(f"Error during async checkpoint saving: {e}")

        # Dispatch background thread
        threading.Thread(
            target=async_save_task,
            args=(
                step_dir,
                model_state_cpu,
                state_dict,
                weights_path,
                state_path,
                self.config.train.hf_repo_id,
                self.config.train.hf_token,
                self.global_step
            )
        ).start()

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load training state and parameters from checkpoint."""
        logger.info(f"Loading checkpoint from: {checkpoint_dir}")
        
        # Load parameters
        weights_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            self.raw_model.load_state_dict(state_dict)
            
        # Load optimizer state and step counters
        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            state_dict = torch.load(state_path, map_location=self.device)
            self.global_step = state_dict.get("global_step", 0)
            
            if "optimizer_state" in state_dict and not self.use_deepspeed:
                self.optimizer.load_state_dict(state_dict["optimizer_state"])
            if "scaler_state" in state_dict and self.scaler.is_enabled():
                self.scaler.load_state_dict(state_dict["scaler_state"])
