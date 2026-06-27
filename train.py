"""
CLI entry point to launch pretraining with Picotron.
Usage: python train.py config.yaml
"""

import os
import sys
import torch
import torch.distributed as dist

from picotron.config import load_config_from_yaml
from picotron.models.llama import LLaMAModel
from picotron.data.tokenized_dataset import TokenizedDataset
from picotron.data.dataloader import get_dataloader
from picotron.trainer import Trainer
from picotron.utils.logging import logger

def main() -> None:
    """Load configuration and execute the trainer."""
    if len(sys.argv) < 2:
        print("Usage: python train.py <path_to_config.yaml>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    cfg = load_config_from_yaml(config_path)
    
    # Initialize torch.distributed if launching with torchrun / multiple GPUs
    distributed = False
    rank = 0
    world_size = 1
    
    # Simple check for distributed launch environments
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        distributed = True
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            
    logger.set_rank(rank)
    logger.info("Initializing Picotron...")
    
    # Set random seeds
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)
        
    # Build Model
    logger.info("Building LLaMAModel...")
    model = LLaMAModel(cfg.model)
    
    # Build Dataset and Dataloader
    logger.info(f"Loading dataset from: {cfg.data.dataset_path}")
    dataset = TokenizedDataset(
        bin_path=cfg.data.dataset_path,
        sequence_length=cfg.data.sequence_length
    )
    
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=cfg.data.micro_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=cfg.train.seed,
        shuffle=True
    )
    
    # Initialize Trainer and start training
    trainer = Trainer(config=cfg, model=model, train_dataloader=dataloader)
    logger.info("Starting training loop...")
    trainer.train()
    
    # Cleanup distributed process group if needed
    if distributed:
        dist.destroy_process_group()
        
    logger.info("Picotron training successfully completed!")

if __name__ == "__main__":
    main()
