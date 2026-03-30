import os
import re
import math
import uuid
import random
import string
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import wandb

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    # Method and base configuration
    parser.add_argument("--method", type=str, required=True,
                        choices=['Domain', 'RNC', 'L1'], 
                        help='method to use')
    parser.add_argument('--base_dir', type=str, required=True, help='base directory for experiments')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--img_size', type=int, default=224, help='input image size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, required=True, help='number of training epochs')
    parser.add_argument('--decoder_epochs', type=int, default=None, help='number of training epochs for decoder, use for Domain and RNC method only')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer to use')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--train_ratio', type=float, default=1.0, help='training data ratio')
    parser.add_argument('--w', type=float, default=1.0, help='regularization weight')
    
    # Model and output configuration
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'ViT'])
    parser.add_argument('--model_layer_output', type=str, default=None, help='final layer of encoder in resnet type model (e.g., layer4, layer3, layer2, layer1)')
    parser.add_argument('--output_list', type=str, nargs='+', help='specify output type of each dimension, e.g., [az, el, ip, cls, id ...]')
    parser.add_argument('--output_dimension_list', type=int, nargs='+', help='specify number of eigenvectors correspond to the output. Default is 1 for each output type')
    parser.add_argument("--encoder_loss_list", type=str, nargs='+', help='specify loss function for each output type for encoder, e.g., [l1, RNC, SupCon, ...]')
    parser.add_argument('--decoder_loss_list', type=str, nargs='+', help='specify loss function for each output type for decoder, e.g., [l1, CE, ...]')
    parser.add_argument('--aug', type=str, default='None', help='augmentations')

    # Dataset configuration
    parser.add_argument('--data_folder', type=str, required=False, help='path to dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=['MPIIFaceGazeNormDataset','ShapeNetRenderingDataset','RotationMNISTDataset'], help='dataset')

    parser.add_argument('--data_prefix', type=str, nargs='+', default=None, help='prefix for train / val dataset')
    parser.add_argument('--test_data_prefix', type=str, nargs='+', default=None, help='prefix for test dataset')

    # EncoderLoss Parameters
    # parser.add_argument('--temp', type=float, default=2, help='temperature')
    parser.add_argument('--temp_list', type=float, nargs='+', help='specify temp for different loss of encoder')
    parser.add_argument('--w_list', type=float, nargs='+', help='specify weight for different loss of encoder')
    parser.add_argument('--label_diff', type=str, default='l1', choices=['l1', 'l2'], help='label distance function')
    parser.add_argument('--feature_sim', type=str, default='l2', choices=['l2'], help='feature similarity function')

    # Domain-specific parameters
    parser.add_argument('--no_hungarian_alignment', action='store_true', help='disable hungarian alignment')
    parser.add_argument("--max_features_for_eigen_update", type=int, default=None, help="max number of features to use for updating eigenvectors")
    parser.add_argument("--max_samples_for_model_update", type=int, default=None, help="max number of features to use for updating eigenvectors")
    parser.add_argument("--fix_eigenvector_epoch", type=int, default=None, help="max epoch to fix eigenvector calculation")
    parser.add_argument("--normalize_classification", action='store_true', default=False, help="normalize classification output")

    # Checkpoint and resume
    parser.add_argument('--ckpt', type=str, default=None, help='path to the trained encoder')
    parser.add_argument('--ckpt_regressor', type=str, default=None, help='path to the trained regressor')
    parser.add_argument('--resume', action='store_true', help='resume training from a checkpoint')
    parser.add_argument('--wandb_id', type=str, default=None,
                       help='Resume from wandb run ID - will automatically find checkpoints and continue training')

    # Logging and debugging
    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--save_curr_freq', type=int, default=50, help='save curr last frequency')
    parser.add_argument('--eval_freq', type=int, default=1, help='evaluation frequency')
    parser.add_argument('--trial', type=str, default='0', help='experiment identifier')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--entity', type=str, default='ngocbach-arizona-state-university', help='wandb entity (team) name')

    opt = parser.parse_args()
    
    opt.output_dim = len(opt.output_list)  # Number of outputs
    opt.classification_keywords = ['cls', 'class', 'classification', 'category', 'label', 'id']  # Keywords to identify classification outputs, specify in output_list

    # If resuming from wandb ID, skip the normal model path setup
    if not opt.wandb_id:
        opt.model_path = f'{opt.base_dir}/experiments/{opt.dataset}/{opt.method}/'
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:6]
        opt.model_name = f"{timestamp}_{random_id}"
        
        if opt.debug:
            print("Debug mode enabled")
            # Change save freq, epoch to low
            opt.epochs = 2
            opt.decoder_epochs = 2
            opt.save_freq = 1
            opt.save_curr_freq = 1
            opt.trial = "debug_" + opt.trial
            opt.num_workers = 8
            opt.batch_size = 256
            opt.learning_rate = 1e-4
            opt.model_path = f'{opt.base_dir}/experiments/debug/{opt.dataset}/{opt.method}'
            opt.model_name = "debug_" + opt.model_name

        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        os.makedirs(opt.save_folder, exist_ok=True)

        print(f"Model name: {opt.model_name}")
    else:
        print(f"Resume mode: will search for wandb ID {opt.wandb_id}")
    
    print(f"Options: {opt}")
    
    return opt

def get_latest_checkpoint_info(save_folder, method):
    """
    Determine what checkpoints exist and their latest epochs.
    Returns dict with checkpoint info based on method type.
    """
    info = {
        'has_encoder': False,
        'has_regressor': False,
        'has_joint': False,
        'latest_encoder_epoch': 0,
        'latest_regressor_epoch': 0,
        'latest_joint_epoch': 0,
        'encoder_path': None,
        'regressor_path': None,
        'joint_path': None
    }
    
    if not os.path.exists(save_folder):
        return info
    
    files = os.listdir(save_folder)
    
    if method in ['Domain', 'RNC']:
        # Handle separate encoder/regressor checkpoints
        # Check for encoder checkpoints
        encoder_files = [f for f in files if f.startswith('ckpt_epoch_') and f.endswith('.pth')]
        if encoder_files:
            epochs = [int(f.split('_')[2].split('.')[0]) for f in encoder_files]
            info['latest_encoder_epoch'] = max(epochs)
            info['encoder_path'] = os.path.join(save_folder, f'ckpt_epoch_{info["latest_encoder_epoch"]}.pth')
            info['has_encoder'] = True
        elif 'last_encoder.pth' in files:
            info['encoder_path'] = os.path.join(save_folder, 'last_encoder.pth')
            info['has_encoder'] = True
            # Load to get epoch info
            try:
                ckpt = torch.load(info['encoder_path'], map_location='cpu')
                info['latest_encoder_epoch'] = ckpt.get('epoch', 0)
            except:
                info['latest_encoder_epoch'] = 0
        
        # Check for regressor checkpoints
        regressor_files = [f for f in files if f.startswith('regressor_epoch_') and f.endswith('.pth')]
        if regressor_files:
            epochs = [int(f.split('_')[2].split('.')[0]) for f in regressor_files]
            info['latest_regressor_epoch'] = max(epochs)
            info['regressor_path'] = os.path.join(save_folder, f'regressor_epoch_{info["latest_regressor_epoch"]}.pth')
            info['has_regressor'] = True
        elif 'last_regressor.pth' in files:
            info['regressor_path'] = os.path.join(save_folder, 'last_regressor.pth')
            info['has_regressor'] = True
            # Load to get epoch info
            try:
                ckpt = torch.load(info['regressor_path'], map_location='cpu')
                info['latest_regressor_epoch'] = ckpt.get('epoch', 0)
            except:
                info['latest_regressor_epoch'] = 0
    
    elif method == 'L1':
        # Handle joint model checkpoints
        joint_files = [f for f in files if f.startswith('joint_ckpt_epoch_') and f.endswith('.pth')]
        if joint_files:
            epochs = [int(f.split('_')[3].split('.')[0]) for f in joint_files]
            info['latest_joint_epoch'] = max(epochs)
            info['joint_path'] = os.path.join(save_folder, f'joint_ckpt_epoch_{info["latest_joint_epoch"]}.pth')
            info['has_joint'] = True
        elif 'last_joint.pth' in files:
            info['joint_path'] = os.path.join(save_folder, 'last_joint.pth')
            info['has_joint'] = True
            # Load to get epoch info
            try:
                ckpt = torch.load(info['joint_path'], map_location='cpu')
                info['latest_joint_epoch'] = ckpt.get('epoch', 0)
            except:
                info['latest_joint_epoch'] = 0
    
    return info

def load_config_from_wandb(wandb_id, opt):
    """
    Load the original configuration from a wandb run and update opt.
    Automatically detects all parameters from opt without manual maintenance.
    """
    try:
        # Initialize wandb API
        api = wandb.Api()
        
        # Get the run - you may need to adjust the entity/project
        run_path = f"ngocbach-arizona-state-university/Domain Expansion/{wandb_id}"
        run = api.run(run_path)
        
        # Get the config from the original run
        original_config = run.config
        
        # Automatically get all current parameters from opt
        current_params = set(vars(opt).keys())
        
        # Parameters that should be treated as boolean flags
        boolean_params = {
            'no_hungarian_alignment', 'resume', 'debug', 
            'normalize_classification'
        }
        
        # Parameters that should be treated as lists
        list_params = {
            'data_prefix', 'test_data_prefix', 'output_list', 
            'output_dimension_list', 'encoder_loss_list', 'decoder_loss_list'
        }
        
        loaded_params = []
        skipped_params = []
        
        # Update opt with all matching parameters from wandb config
        for param in current_params:
            if param in original_config:
                try:
                    value = original_config[param]
                    
                    # Handle different parameter types
                    if param in boolean_params:
                        # Handle boolean conversion
                        if isinstance(value, str):
                            value = value.lower() == 'true'
                        else:
                            value = bool(value)
                    
                    elif param in list_params:
                        # Handle list conversion
                        if isinstance(value, str):
                            # If stored as string representation, try to parse it
                            if value.startswith('[') and value.endswith(']'):
                                try:
                                    value = eval(value)  # Safe for simple lists
                                except:
                                    # Fallback: split by comma and strip
                                    value = [item.strip().strip("'\"") for item in value[1:-1].split(',') if item.strip()]
                            else:
                                value = [value]  # Single item becomes list
                        elif not isinstance(value, list):
                            value = [value]  # Convert single value to list
                    
                    else:
                        # For other types, try to match the current type in opt
                        current_value = getattr(opt, param, None)
                        if current_value is not None and not isinstance(current_value, type(None)):
                            try:
                                # Try to convert to the same type as current value
                                if isinstance(current_value, (int, float, str)):
                                    value = type(current_value)(value)
                            except (ValueError, TypeError):
                                # If conversion fails, use the wandb value as-is
                                pass
                    
                    setattr(opt, param, value)
                    loaded_params.append(param)
                    print(f"Loaded {param}: {value}")
                    
                except Exception as e:
                    print(f"Warning: Could not load parameter {param}={original_config[param]}: {e}")
                    skipped_params.append(param)
        
        # Check for parameters that exist in wandb but not in current opt
        wandb_only_params = set(original_config.keys()) - current_params
        if wandb_only_params:
            print(f"Parameters in wandb config but not in current opt (will be ignored): {sorted(wandb_only_params)}")
        
        # Check for parameters in current opt that weren't in wandb
        current_only_params = current_params - set(original_config.keys())
        if current_only_params:
            print(f"Parameters in current opt but not in wandb config (keeping current values): {sorted(current_only_params)}")
        
        # Reconstruct save_folder if model_name and model_path are available
        if 'model_name' in original_config and 'model_path' in original_config:
            opt.model_name = original_config['model_name']
            opt.model_path = original_config['model_path']
            opt.save_folder = os.path.join(opt.model_path, opt.model_name)
            print(f"Constructed save_folder from wandb config: {opt.save_folder}")
        else:
            # Fallback if config did not store those keys
            opt.save_folder = os.path.join(opt.base_dir, opt.dataset, opt.method, "default")
            print(f"Fallback save_folder: {opt.save_folder}")
        
        print(f"Successfully loaded {len(loaded_params)} parameters from wandb run")
        if skipped_params:
            print(f"Skipped parameters due to errors: {skipped_params}")
        
        return opt
        
    except Exception as e:
        print(f"Warning: Could not load config from wandb run {wandb_id}: {e}")
        print("Proceeding with current configuration...")
        return opt

def resume_from_wandb_id(opt, wandb_id):
    """
    Resume training from a wandb run ID with integrated wandb setup.
    Returns updated opt with checkpoint paths and resume info.
    Handles both Domain/RNC (separate encoder/regressor) and L1 (joint model) methods.
    """
    # First, load the configuration from wandb (this will set model_name, model_path, save_folder)
    opt = load_config_from_wandb(wandb_id, opt)
    
    print(f"Using save folder: {opt.save_folder}")
    
    # Get checkpoint info based on method
    ckpt_info = get_latest_checkpoint_info(opt.save_folder, opt.method)
    
    # Check if we have appropriate checkpoints for the method
    if opt.method in ['Domain', 'RNC']:
        if not ckpt_info['has_encoder'] and not ckpt_info['has_regressor']:
            print(f"No encoder/regressor checkpoints found in {opt.save_folder}")
    elif opt.method == 'L1':
        if not ckpt_info['has_joint']:
            print(f"No joint model checkpoints found in {opt.save_folder}")
    
    # Update opt with found paths
    opt.resume = True
    opt.wandb_id = wandb_id
    
    if opt.method in ['Domain', 'RNC']:
        if ckpt_info['has_encoder']:
            opt.ckpt = ckpt_info['encoder_path']
            print(f"Will resume encoder from: {opt.ckpt} (epoch {ckpt_info['latest_encoder_epoch']})")
        
        if ckpt_info['has_regressor']:
            opt.ckpt_regressor = ckpt_info['regressor_path']
            print(f"Will resume regressor from: {opt.ckpt_regressor} (epoch {ckpt_info['latest_regressor_epoch']})")
    
    elif opt.method == 'L1':
        if ckpt_info['has_joint']:
            opt.ckpt = ckpt_info['joint_path']  # Use ckpt for joint model path
            print(f"Will resume joint model from: {opt.ckpt} (epoch {ckpt_info['latest_joint_epoch']})")
    
    # Setup wandb with resume capability - integrated here
    print(f"Resuming wandb run with ID: {opt.wandb_id}")
    wandb.init(
        project="Domain Expansion",  # Adjust project name as needed
        id=opt.wandb_id,
        entity=opt.entity,  # Use the entity from options
        resume="must",
        force=True,  # Force resume even if run is already active
        reinit=True,
        config=vars(opt)
    )
    
    return opt, ckpt_info

def init_wandb(opt):
    """
    Initialize wandb for tracking experiments.
    """
    
    # Start new wandb run for non-resume case
    wandb.init(
        project="Domain Expansion",  # Adjust project name as needed
        config=vars(opt),
        entity=opt.entity,  # Use the entity from options
        name=getattr(opt, 'model_name', None)
    )
    
    opt.wandb_id = wandb.run.id
    print(f"Started new wandb run with ID: {opt.wandb_id}")
    
    return opt