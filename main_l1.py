import argparse
import os
import logging
import time
import torch
import torch.nn as nn
from core.parser_utils import parse_option, resume_from_wandb_id, init_wandb
from core.train_utils import train_joint, test
from core.utils import set_loader, adjust_learning_rate, measure_train_memory
from core.model import L1_Encoder
import wandb

def main():
    opt = parse_option()
    
    # Handle wandb resume by ID
    ckpt_info = None
    if opt.wandb_id:  # Simplified check
        opt, ckpt_info = resume_from_wandb_id(opt, opt.wandb_id)
    else:
        # Start new wandb run for non-resume case
        opt = init_wandb(opt)  # Initialize wandb if no ID is provided
    
    # Define wandb metrics to avoid step conflicts
    wandb.define_metric("regressor/epoch")
    wandb.define_metric("val/epoch")
    wandb.define_metric("test/epoch")
    # Define metric groups with their respective step metrics
    wandb.define_metric("regressor/*", step_sync=False, step_metric="regressor/epoch")
    wandb.define_metric("val/*", step_sync=False, step_metric="val/epoch")
    wandb.define_metric("test/*", step_sync=False, step_metric="test/epoch")
    
    # 1) Build data loaders (train/val/test)
    train_loader, valid_loader, test_loader = set_loader(opt)
    
    # 2) Build L1_Encoder (which already contains its regressor)
    model = L1_Encoder(opt, in_channels=3).cuda()
    
    # 3) Set up a single optimizer over the entire model (encoder + internal regressor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.weight_decay
    )
    
    # 4) Loss: L1Loss for scalar regression
    criterion = nn.L1Loss().cuda()
    
    start_epoch = 1
    
    # 5) Optionally resume from a joint checkpoint
    if opt.ckpt and os.path.isfile(opt.ckpt):
        print(f"Loading joint checkpoint from: {opt.ckpt}")
        try:
            ckpt_data = torch.load(opt.ckpt, map_location='cpu')
            model.load_state_dict(ckpt_data['model_state_dict'])
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
            start_epoch = ckpt_data.get('epoch', 0) + 1
            print(f"Successfully loaded checkpoint from epoch {ckpt_data.get('epoch', 0)}")
            print(f"Resuming training from epoch {start_epoch}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 1
    elif opt.ckpt:
        print(f"Warning: Checkpoint file not found at {opt.ckpt}")
        print("Starting training from scratch...")
    else:
        print("No checkpoint specified, starting training from scratch...")
    
    # 6) Training loop: jointly train encoder + internal regressor
    for epoch in range(start_epoch, opt.epochs + 1):
        
        # Adjust learning rate if needed
        adjust_learning_rate(opt, optimizer, epoch)
        
        # One epoch of joint training
        measure_train_memory(train_joint, train_loader, model, criterion, optimizer, epoch, opt)
        
        # Optional: validate on the validation set
        # Since test expects separate encoder + regressor, we pass model.model (encoder)
        # and model.regressor (the final linear+Tanh block) separately.
        test(valid_loader, model.model, model.regressor, opt, device="cuda", val=True, epoch=epoch)
        
        # Save checkpoints every save_freq epochs
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f'joint_ckpt_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wandb_id': opt.wandb_id,  # Save wandb ID for future resume
                'method': opt.method,
                'dataset': opt.dataset
            }, save_file)
            print(f"Saved joint checkpoint: {save_file}")
        
        # Always update the "last" checkpoint at save_curr_freq intervals
        if epoch % opt.save_curr_freq == 0:
            last_file = os.path.join(opt.save_folder, 'last_joint.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wandb_id': opt.wandb_id,  # Save wandb ID for future resume
                'method': opt.method,
                'dataset': opt.dataset
            }, last_file)
            print(f"Updated last checkpoint: {last_file}")
    
    # Final test
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    test(test_loader, model.model, model.regressor, opt, device="cuda", val=False)
    
    print("Finished joint training.")
    print("="*50)

if __name__ == '__main__':
    main()