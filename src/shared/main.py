import argparse
import yaml
import os
import torch
import lightning as L
from torch.utils.data import DataLoader
import time
import shutil
from diffusers.utils.logging import disable_progress_bar

from shared.data.nabirds_dataset import NABirdsDataset
from shared.callbacks import UnifiedLoggerCallback

from omini.model_wrapper import OminiModelWrapper
from omini.data_utils import OminiBatchWrapper

from insert.model_wrapper import InsertAnythingWrapper
from insert.data_utils import InsertBatchWrapper

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_dataset_and_model(config):
    """
    Factory function: Instantiates the correct Model and Data Wrapper
    based on 'model_type' in the config.
    """
    model_type = config.get("model_type")
    
    if model_type == "insert":
        # Insert Anything: Uses Expansion + Probabilistic Masks
        expand_crop = True
        prob_mask = True
        mask_background = True
    else:
        # Omini: Uses Tight Crops + GT Masks (Original Behavior)
        expand_crop = False
        prob_mask = False
        mask_background = False
    
    base_dataset = NABirdsDataset(
        root=config['train']['dataset']['root'],
        split='train',
        condition_size=tuple(config['train']['dataset']['condition_size']),
        target_size=tuple(config['train']['dataset']['target_size']),
        return_raw=False, 
        augment=config['train']['dataset'].get('augment', False),
        expand_crop=expand_crop,
        probabilistic_mask=prob_mask,
        mask_background=mask_background,
        **{k: v for k, v in config['train']['dataset'].items() if k not in [
            'root', 'condition_size', 'target_size', 'augment'
        ]}
    )

    print(f"--- Initialized Base Dataset with {len(base_dataset)} pairs ---")

    if model_type == "omini":
        print("--- Mode: OminiControl ---")
        train_dataset = OminiBatchWrapper(base_dataset, config)
        model = OminiModelWrapper(config)
        
    elif model_type == "insert":
        print("--- Mode: Insert Anything ---")
        train_dataset = InsertBatchWrapper(base_dataset, config)
        model = InsertAnythingWrapper(config)
        
    else:
        raise ValueError(f"Invalid model_type: '{model_type}'. Must be 'omini' or 'insert'.")
        
    return train_dataset, model

def main():
    disable_progress_bar()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--job_dir", type=str, required=True, help="Single output directory for everything")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in job_dir")
    args = parser.parse_args()

    # 1. Setup Directory
    os.makedirs(args.job_dir, exist_ok=True)
    os.makedirs(os.path.join(args.job_dir, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(args.job_dir, "samples"), exist_ok=True)
    
    # 2. Config Management
    config_save_path = os.path.join(args.job_dir, "config.yaml")
    if os.path.exists(config_save_path) and args.resume:
        # Load from existing run
        print(f"Resuming from {args.job_dir}...")
        config = load_config(config_save_path)
    else:
        # New Run: Load from arg and save copy
        config = load_config(args.config)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        
    print(args)
    print(config)

    # 3. Resume Logic
    resume_step_offset = 0
    ckpt_path = None
    if args.resume:
        ckpt_dir = os.path.join(args.job_dir, "ckpt")
        checkpoints = [d for d in os.listdir(ckpt_dir) if d.startswith("step_")]
        if checkpoints:
            latest_step = max([int(d.split("_")[1]) for d in checkpoints])
            ckpt_path = os.path.join(ckpt_dir, f"step_{latest_step}")
            resume_step_offset = latest_step
            print(f"Found checkpoint: {ckpt_path}, resuming from step count: {resume_step_offset}")
    
    total_max_steps = config['train'].get("max_steps", 10000)
    current_run_steps = total_max_steps - resume_step_offset

    if current_run_steps <= 0:
        print(f"Training already finished! (Target: {total_max_steps}, Current: {resume_step_offset})")
        return

    L.seed_everything(args.seed)
    train_dataset, model = get_dataset_and_model(config)

    print("--- Setting up Visualization Indices ---")
    # Create the validation dataset (Raw access, no wrappers needed here)
    val_dataset = NABirdsDataset(
        root=config['train']['dataset']['root'],
        split='test',
        condition_size=tuple(config['train']['dataset']['condition_size']),
        target_size=tuple(config['train']['dataset']['target_size']),
        return_raw=False, 
        augment=False, # Important: False ensures consistent, clean evaluation
        **{k: v for k, v in config['train']['dataset'].items() if k not in [
            'root', 'condition_size', 'target_size', 'augment', 'feature_type', 'top_k'
        ]}
    )

    # Define fixed indices for consistency (e.g., 8 evenly spaced samples)
    # This replaces the complex pre-sampling logic
    num_vis_samples = 8
    total_len = len(val_dataset)
    # Using torch.linspace to get evenly spaced indices, cast to int list
    vis_indices = torch.linspace(0, total_len - 1, num_vis_samples).int().tolist()
    
    print(f"--- Tracking Indices: {vis_indices} ---")

    # DataLoader (for Training)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train'].get('batch_size', 1),
        shuffle=True,
        num_workers=config['train'].get('dataloader_workers', 4),
        pin_memory=True
    )

    # Callbacks
    callbacks = [
        UnifiedLoggerCallback(
            save_path=args.job_dir, 
            config=config, 
            val_dataset=val_dataset,
            vis_indices=vis_indices,
            resume_step_offset=resume_step_offset
        )
    ]

    trainer = L.Trainer(
        default_root_dir=args.job_dir,
        callbacks=callbacks,
        max_steps=current_run_steps,
        accumulate_grad_batches=config['train'].get("accumulate_grad_batches", 1),
        precision=config.get("precision", "bf16"),
        gradient_clip_val=config['train'].get("gradient_clip_val", 1.0),
        accelerator="gpu",
        devices="auto",
        strategy="auto", # Handles DDP automatically if multiple GPUs detected
        enable_checkpointing=False, # We handle this in UnifiedLoggerCallback
        logger=False, # We handle this in UnifiedLoggerCallback
        enable_progress_bar=False,
    )

    print(f"--- Starting Training: Job {args.job_dir} ---")
    
    # Start Training
    # Note: We don't pass ckpt_path for resume logic usually with LoRA 
    # unless using full state restore. Our Wrapper handles LoRA loading in __init__.
    if ckpt_path:
        model.load_lora(ckpt_path) # Ensure wrappers have this method (alias for load_lora_weights)

    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()