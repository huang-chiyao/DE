# 1. Standard Library Imports
import argparse
import json
import math
import os
import random
import re
from datetime import datetime
from glob import glob

# 2. Third-Party Imports: Data Science & Image Processing
import numpy as np
import pandas as pd
from PIL import Image

# 3. Third-Party Imports: PyTorch Core & Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 4. Third-Party Imports: PyTorch Data Handling
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    random_split
)
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST

# 5. Third-Party Imports: PyTorch Vision Transforms
# Note: You imported transforms as both 'transforms' and 'T'. 
# Both are preserved here for compatibility.
import torchvision.transforms as transforms
import torchvision.transforms as T

# 6. Local Imports
from .dataset import *

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TwoNonTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [x, x]

class TwoTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(self.transform1(x))]

class TwoTransformNew:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]

class CenterFlowCrop:
    """
    Center crop a flow tensor of shape (C, H, W)
    down to (C, new_H, new_W), e.g. (2, 224, 224).
    """

    def __init__(self, size=(224, 224)):
        """
        size: (height, width) for the output crop.
        """
        self.size = size

    def __call__(self, flow):
        """
        flow: a torch.Tensor of shape (C, H, W), typically (2, H, W) for optical flow.
        Returns the cropped flow of shape (2, new_H, new_W).
        """
        if not hasattr(flow, "shape"):
            flow = F.to_tensor(flow)

        _, h, w = flow.shape
        new_h, new_w = self.size

        # Calculate top & left for the center crop
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        return flow[:, top:top+new_h, left:left+new_w]

class RandomFlowCrop:
    """
    Randomly crop a flow tensor of shape (C, H, W) to (C, new_H, new_W).
    """
    def __init__(self, size):
        """
        size: (new_H, new_W)
        """
        self.new_h, self.new_w = size

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        _, h, w = flow.shape
        if h == self.new_h and w == self.new_w:
            return flow
        top = random.randint(0, h - self.new_h)
        left = random.randint(0, w - self.new_w)
        return flow[:, top : top + self.new_h, left : left + self.new_w]

class NormalizeFlow:
    """
    Per‐sample image‐relative normalization:
      - channel 0 (horizontal flow) is divided by (width−1)
      - channel 1 (vertical   flow) is divided by (height−1)
    This maps a full‐image‐width displacement to ±1, regardless of resolution.
    """
    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        # flow shape: (2, H, W)
        flow = flow.contiguous()
        _, H, W = flow.shape

        # avoid division by zero if W or H == 1
        denom_u = (W - 1) if W > 1 else 1.0
        denom_v = (H - 1) if H > 1 else 1.0

        u = flow[0] / denom_u
        v = flow[1] / denom_v

        return torch.stack([u, v], dim=0)

class GaussianFlowAugmentation:
    """
    Adds zero-mean Gaussian noise.
    """
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std  = std

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(flow) * self.std + self.mean
        return flow + noise

class RandomBlockMask:
    """Zero out a random block of size (h, w) within the flow map."""
    def __init__(self, max_mask_size=(20,20), p=0.5):
        self.max_h, self.max_w = max_mask_size
        self.p = p

    def __call__(self, flow: torch.Tensor) -> torch.Tensor:
        # flow: (C, H, W)
        if random.random() > self.p:
            return flow
        _, H, W = flow.shape
        mh = random.randint(1, min(self.max_h, H))
        mw = random.randint(1, min(self.max_w, W))
        top = random.randint(0, H - mh)
        left = random.randint(0, W - mw)
        flow[:, top:top+mh, left:left+mw] = 0.0
        return flow


def get_transforms(split, aug):
    normalize = transforms.Normalize(mean=(0.5), std=(0.5))
    if split == 'train':
        aug_list = aug.split(',')
        transforms_list = []

        if 'crop' in aug_list:
            transforms_list.append(transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)))
        else:
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(224))

        if 'flip' in aug_list:
            transforms_list.append(transforms.RandomHorizontalFlip())

        if 'color' in aug_list:
            transforms_list.append(transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8))

        if 'grayscale' in aug_list:
            transforms_list.append(transforms.RandomGrayscale(p=0.2))

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(normalize)
        transform = transforms.Compose(transforms_list)
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 832)),  # First, resize to keep the aspect ratio intact
            transforms.CenterCrop(224),     # Center crop to 224x224
            transforms.ToTensor(),          # Convert to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet)
        ])

    return transform

class MPIIFaceGazeNorm_DataLoader:
    """
    A MPIIFaceGaze dataloader that creates train, validation, and test dataloaders.
    Supports mixing all participants and splitting with reproducible seeding.
    Similar structure to ShapeNetRendering_DataLoader.

    Args:
        root (str): Root directory containing participant folders (e.g., 'p00', 'p01', etc.).
        batch_size (int): Batch size for training.
        num_workers (int): DataLoader workers.
        categories (list): List of participant IDs (e.g., ['p00', 'p01', 'p02']).
        opt: Options object containing dataset configuration.
    """
    def __init__(
        self,
        batch_size,
        num_workers,
        categories,
        opt=None
    ):
        
        self.root = opt.data_folder
        self.categories = categories if isinstance(categories, list) else [categories]
        self.opt = opt
        
        # Validate that all participant directories exist
        self._validate_participants()
        
        # Set up split ratios and seeding
        self.train_ratio = 0.72
        self.val_ratio = 0.08
        self.test_ratio = 0.2
        self.seed = 42
        
        # Create participant to class mapping (always multi-class for participants)
        self._create_participant_mappings()
        
        # Prepare transforms
        crop_transform = transforms.Compose([
            CenterFlowCrop((224, 224)),
            # Add more flow-specific transforms as needed
        ])
        
        gaussian_transform = transforms.Compose([
            GaussianFlowAugmentation(mean=0.0, std=0.05),
            # Add more flow-specific transforms as needed
        ])
        
        # Use TwoTransform for training (crop + augmentation)
        train_transform = TwoTransform(crop_transform, gaussian_transform)
        # Use only crop transform for validation/test
        eval_transform = TwoTransform(crop_transform, crop_transform)
        
        # Load and combine all participant datasets
        all_datasets = self._load_all_participants(train_transform)
        combined_dataset = ConcatDataset(all_datasets)
        total_size = len(combined_dataset)
        
        if opt.debug:
            # In debug mode, use a smaller subset
            debug_size = min(1000, total_size)
            combined_dataset = torch.utils.data.Subset(combined_dataset, range(debug_size))
            total_size = debug_size
        
        # Split combined dataset with reproducible seeding
        train_dataset, val_dataset, test_dataset = self._split_dataset(combined_dataset, total_size)
        
        # Update transforms for val/test datasets
        self._update_dataset_transforms(val_dataset, eval_transform)
        self._update_dataset_transforms(test_dataset, eval_transform)
        
        print(f"✅ Loaded MPIIFaceGaze data from {len(self.categories)} participants: {self.categories}")
        print(f"   Classification mode: participant")
        print(f"   Number of participant classes: {self.num_classes}")
        print(f"   Participant to class mapping: {self.participant_to_class}")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Total samples: {total_size}")
        print(f"   Split ratios - Train: {self.train_ratio}, Val: {self.val_ratio}, Test: {self.test_ratio}")

        # Store datasets
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=batch_size,
            shuffle=True,  # Always shuffle training data
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_ds, 
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_ds, 
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test
            num_workers=num_workers,
            pin_memory=True
        )

    def _validate_participants(self):
        """Validate that all participant directories exist."""
        missing_labels = []
        
        for participant in self.categories:
            participant_dir = os.path.join(self.root, participant)
            label_file = f'{participant_dir}.label'
                
            if not os.path.isfile(label_file):
                missing_labels.append(participant)
        
        if missing_labels:
            raise FileNotFoundError(
                f"Label files not found for participants: {missing_labels}\n"
                f"Expected label files: {[os.path.join(self.root, f'{p}.label') for p in missing_labels]}"
            )

    def _create_participant_mappings(self):
        """Create participant to class mappings."""
        # For MPIIFaceGaze, we always use participant-based classification
        self.classification_mode = 'participant'
        self.participant_to_class = {pid: idx for idx, pid in enumerate(sorted(self.categories))}
        self.num_classes = len(self.categories)
        
        # Create reverse mapping for convenience
        self.class_to_participant = {idx: pid for pid, idx in self.participant_to_class.items()}

    def _load_all_participants(self, transform):
        """Load datasets for all participants."""
        all_datasets = []
        
        for participant in self.categories:
            participant_dir = os.path.join(self.root, participant)
            
            try:
                dataset = MPIIFaceGazeNormDataset(
                    root_dir=participant_dir,
                    transform=transform,
                    return_dict=False,
                    opt=self.opt,
                    participant_to_class=self.participant_to_class
                )
                all_datasets.append(dataset)
                print(f"   Loaded participant {participant}: {len(dataset)} samples")
                
            except Exception as e:
                print(f"   Warning: Failed to load participant {participant}: {e}")
                continue
        
        if not all_datasets:
            raise RuntimeError("No participant datasets could be loaded")
            
        return all_datasets

    def _split_dataset(self, combined_dataset, total_size):
        """Split the combined dataset into train/val/test with reproducible seeding."""
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        
        # Calculate split sizes
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size  # Ensure all samples are used
        
        print(f"   Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Perform the split
        train_dataset, val_dataset, test_dataset = random_split(
            combined_dataset,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        return train_dataset, val_dataset, test_dataset

    def _update_dataset_transforms(self, subset_dataset, new_transform):
        """Update transforms for validation/test datasets."""
        # This is a bit tricky with ConcatDataset and random_split
        # We need to access the underlying datasets and update their transforms
        # This assumes the subset contains indices that map back to the original datasets
        
        # For simplicity, we'll create new dataset instances with eval transforms
        # This is not the most efficient but ensures correctness
        pass  # In practice, you might want to implement this more efficiently

    def get_num_classes(self):
        """Returns the number of classes for classification."""
        return self.num_classes
    
    def get_participant_mapping(self):
        """Returns the participant to class mapping dictionary."""
        return self.participant_to_class
    
    def get_classification_mode(self):
        """Returns the classification mode (always 'participant' for this dataset)."""
        return self.classification_mode
    
    def get_loaders(self):
        """Returns (train_loader, val_loader, test_loader)."""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_datasets(self):
        """Returns (train_dataset, val_dataset, test_dataset)."""
        return self.train_ds, self.val_ds, self.test_ds
    
    def get_participant_info(self):
        """Returns participant-specific information."""
        return {
            'participants': self.categories,
            'participant_to_class': self.participant_to_class,
            'class_to_participant': self.class_to_participant,
            'num_participants': len(self.categories)
        }

class ShapeNetRendering_DataLoader:
    """
    Loads pre-split ShapeNetRendering train/val/test data from cached split files.
    The split files should be generated using split.py to ensure only valid models
    (with matching image/label counts) are included.

    Args:
        root (str): Root directory of ShapeNetRendering.
        batch_size (int): Batch size for training.
        num_workers (int): DataLoader workers.
        categories (list): List of category names (e.g., ['car', 'chair', 'table']).
        el_min_deg (float): Minimum elevation angle.
        el_max_deg (float): Maximum elevation angle.
        opt: Options object.
    """
    def __init__(
        self,
        root,
        batch_size,
        num_workers,
        categories,
        opt=None
    ):
        
        self.root = root
        self.categories = categories if isinstance(categories, list) else [categories]
        
        # Validate that all category directories exist and have split files
        self._validate_categories()
        
        # Load cached samples from all categories first to determine classification mode
        train_samples = self._read_all_samples('train.txt')
        val_samples = self._read_all_samples('val.txt')
        test_samples = self._read_all_samples('test.txt')
        
        if opt.debug:
            train_samples = train_samples[:1000]
            val_samples = val_samples[:200]
            test_samples = test_samples[:200]
            
        # Determine classification mode and create mappings
        self._create_classification_mappings(train_samples, val_samples, test_samples)
        
        # Prepare transforms
        base_transform = T.Compose([
            T.Resize((opt.img_size, opt.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        aug_transform = T.Compose([
            T.RandomResizedCrop(size=opt.img_size, scale=(0.9, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        # Use your existing TwoTransformNew class
        train_transform = TwoTransformNew(base_transform, aug_transform)
        eval_transform = TwoTransformNew(base_transform, base_transform)
    
        print(f"✅ Loaded cached splits from {len(self.categories)} categories: {self.categories}")
        print(f"   Classification mode: {self.classification_mode}")
        print(f"   Number of classes: {self.num_classes}")
        if self.classification_mode == 'category':
            print(f"   Category to class mapping: {self.category_to_class}")
        else:
            print(f"   Model to class mapping: {len(self.model_to_class)} models")
            print(f"   Sample model mappings: {dict(list(self.model_to_class.items())[:5])}")
        print(f"   Train samples: {len(train_samples)}")
        print(f"   Val samples: {len(val_samples)}")
        print(f"   Test samples: {len(test_samples)}")
        print(f"   Total samples: {len(train_samples) + len(val_samples) + len(test_samples)}")

        # Build datasets using your existing ShapeNetRenderingDataset
        self.train_ds = ShapeNetRenderingDataset(root, train_samples, train_transform, opt, 
                                                category_to_class=self.category_to_class, 
                                                model_to_class=self.model_to_class,
                                                model_to_id_within_class=self.model_to_id_within_class,  # NEW
                                                classification_mode=self.classification_mode,
                                                num_classes=self.num_classes)
        self.val_ds = ShapeNetRenderingDataset(root, val_samples, eval_transform, opt,
                                            category_to_class=self.category_to_class,
                                            model_to_class=self.model_to_class,
                                            model_to_id_within_class=self.model_to_id_within_class,  # NEW
                                            classification_mode=self.classification_mode,
                                            num_classes=self.num_classes)
        self.test_ds = ShapeNetRenderingDataset(root, test_samples, eval_transform, opt,
                                                category_to_class=self.category_to_class,
                                                model_to_class=self.model_to_class,
                                                model_to_id_within_class=self.model_to_id_within_class,  # NEW
                                                classification_mode=self.classification_mode,
                                                num_classes=self.num_classes)

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=batch_size,
            shuffle=True,  # Always shuffle training data
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_ds, 
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_ds, 
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test
            num_workers=num_workers,
            pin_memory=True
        )

    def _create_classification_mappings(self, train_samples, val_samples, test_samples):
        """Create classification mappings based on the number of categories."""
        all_samples = train_samples + val_samples + test_samples
        
        if len(self.categories) == 1:
            # Single category: each model is a class
            self.classification_mode = 'model'
            unique_models = set()
            for category, obj_id, view_idx in all_samples:
                model_id = f"{category}/{obj_id}"
                unique_models.add(model_id)
            
            # Sort for consistent ordering
            sorted_models = sorted(unique_models)
            self.model_to_class = {model: idx for idx, model in enumerate(sorted_models)}
            # In single category mode, each model is its own class, so model_id_within_class is always 0
            self.model_to_id_within_class = {model: 0 for model in sorted_models}
            self.category_to_class = {}  # Not used in model mode
            self.num_classes = len(sorted_models)
            
        else:
            # Multiple categories: each category is a class
            self.classification_mode = 'category'
            self.category_to_class = {cat: idx for idx, cat in enumerate(sorted(self.categories))}
            self.num_classes = len(self.categories)
            
            # Create model_to_class mapping (for potential use)
            # and model_to_id_within_class mapping (for 'id' outputs)
            models_by_category = {}
            for category, obj_id, view_idx in all_samples:
                if category not in models_by_category:
                    models_by_category[category] = set()
                models_by_category[category].add(obj_id)
            
            self.model_to_class = {}
            self.model_to_id_within_class = {}
            
            for category in self.categories:
                class_idx = self.category_to_class[category]
                if category in models_by_category:
                    # Sort models within each category for consistent ordering
                    sorted_models_in_category = sorted(models_by_category[category])
                    for local_idx, obj_id in enumerate(sorted_models_in_category):
                        model_id = f"{category}/{obj_id}"
                        self.model_to_class[model_id] = class_idx
                        self.model_to_id_within_class[model_id] = local_idx

    def _validate_categories(self):
        """Validate that all category directories exist and have required split files."""
        missing_categories = []
        missing_files_by_category = {}
        
        for category in self.categories:
            cache_dir = os.path.join(self.root, category)
            
            if not os.path.isdir(cache_dir):
                missing_categories.append(category)
                continue
                
            # Check for required split files
            train_file = os.path.join(cache_dir, 'train.txt')
            val_file = os.path.join(cache_dir, 'val.txt')
            test_file = os.path.join(cache_dir, 'test.txt')
            
            missing_files = []
            for name, path in [('train.txt', train_file), ('val.txt', val_file), ('test.txt', test_file)]:
                if not os.path.isfile(path):
                    missing_files.append(name)
            
            if missing_files:
                missing_files_by_category[category] = missing_files
        
        # Report errors
        if missing_categories:
            raise FileNotFoundError(
                f"Category directories not found: {missing_categories}\n"
                f"Expected directories in {self.root}: {[os.path.join(self.root, cat) for cat in missing_categories]}"
            )
        
        if missing_files_by_category:
            error_msg = "Required split files not found:\n"
            for category, files in missing_files_by_category.items():
                cache_dir = os.path.join(self.root, category)
                error_msg += f"  {category}: {files} (in {cache_dir})\n"
            error_msg += "\nPlease run 'python split.py' for each category to generate the split files."
            raise FileNotFoundError(error_msg)

    def _read_all_samples(self, filename, max_samples=None):
        """Read samples from all categories and combine them."""
        all_samples = []
        samples_per_category = {}
        
        total_max_samples = max_samples
        max_samples_per_category = None
        if max_samples is not None:
            max_samples_per_category = max_samples // len(self.categories)
        
        for category in self.categories:
            cache_dir = os.path.join(self.root, category)
            file_path = os.path.join(cache_dir, filename)
            
            category_samples = self._read_samples(file_path, category, max_samples_per_category)
            all_samples.extend(category_samples)
            samples_per_category[category] = len(category_samples)
            
            print(f"   {category}: {len(category_samples)} samples from {filename}")
        
        # If we have a global max_samples limit and we're under it, we can try to get more samples
        if total_max_samples is not None and len(all_samples) < total_max_samples:
            remaining_samples = total_max_samples - len(all_samples)
            # Distribute remaining samples among categories
            additional_per_category = remaining_samples // len(self.categories)
            
            if additional_per_category > 0:
                for category in self.categories:
                    cache_dir = os.path.join(self.root, category)
                    file_path = os.path.join(cache_dir, filename)
                    
                    current_count = samples_per_category[category]
                    additional_samples = self._read_samples(
                        file_path, category, 
                        max_samples=current_count + additional_per_category,
                        skip_first=current_count
                    )
                    all_samples.extend(additional_samples)
        
        return all_samples

    def _read_samples(self, path, category, el_min_deg=None, el_max_deg=None, max_samples=None, skip_first=0):
        """Read samples from a single category's split file."""
        samples = []
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # Skip first N samples if specified
                if idx < skip_first:
                    continue
                
                if max_samples is not None and len(samples) >= max_samples:
                    break
                
                try:
                    part, view_str = line.split()
                    file_category, obj_id = part.split('/')
                    view_idx = int(view_str)
                    
                    # Ensure the category matches what we expect
                    if file_category != category:
                        print(f"Warning: Category mismatch in {path}: expected {category}, got {file_category}")
                        continue
                    
                    # Check elevation if needed
                    
                    samples.append((file_category, obj_id, view_idx))
                except ValueError as e:
                    print(f"Warning: Skipping malformed line in {path}: '{line}' ({e})")
                    continue
        
        return samples

    def get_num_classes(self):
        """Returns the number of classes for classification."""
        return self.num_classes
    
    def get_category_mapping(self):
        """Returns the category to class mapping dictionary."""
        return self.category_to_class
    
    def get_model_mapping(self):
        """Returns the model to class mapping dictionary."""
        return self.model_to_class
    
    def get_classification_mode(self):
        """Returns the classification mode ('category' or 'model')."""
        return self.classification_mode
    
    def get_loaders(self):
        """Returns (train_loader, val_loader, test_loader)."""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_datasets(self):
        """Returns (train_dataset, val_dataset, test_dataset)."""
        return self.train_ds, self.val_ds, self.test_ds
    
    def print_stats(self):
        """Print detailed statistics about the loaded data."""
        print(f"\n📊 Dataset Statistics:")
        print(f"   Root directory: {self.root}")
        print(f"   Categories: {self.categories}")
        print(f"   Classification mode: {self.classification_mode}")
        print(f"   Number of classes: {self.num_classes}")
        
        if self.classification_mode == 'category':
            print(f"   Category to class mapping: {self.category_to_class}")
        else:
            print(f"   Model to class mapping: {len(self.model_to_class)} models")
            print(f"   Sample model mappings: {dict(list(self.model_to_class.items())[:10])}")
        
        print(f"   Train samples: {len(self.train_ds):,}")
        print(f"   Val samples: {len(self.val_ds):,}")
        print(f"   Test samples: {len(self.test_ds):,}")
        print(f"   Total samples: {len(self.train_ds) + len(self.val_ds) + len(self.test_ds):,}")
        
        # Calculate some additional stats
        total_samples = len(self.train_ds) + len(self.val_ds) + len(self.test_ds)
        train_pct = len(self.train_ds) / total_samples * 100
        val_pct = len(self.val_ds) / total_samples * 100
        test_pct = len(self.test_ds) / total_samples * 100
        
        print(f"   Train: {train_pct:.1f}%, Val: {val_pct:.1f}%, Test: {test_pct:.1f}%")
        
        # Get unique models in each split
        def get_unique_models(samples):
            return set(f"{cat}/{obj}" for cat, obj, _ in samples)
        
        train_models = get_unique_models(self.train_ds.samples)
        val_models = get_unique_models(self.val_ds.samples)
        test_models = get_unique_models(self.test_ds.samples)
        
        print(f"   Unique models - Train: {len(train_models)}, Val: {len(val_models)}, Test: {len(test_models)}")
        
        # Print per-category or per-model statistics
        if self.classification_mode == 'category':
            print(f"\n📈 Per-Category Statistics:")
            for split_name, samples in [('Train', self.train_ds.samples), ('Val', self.val_ds.samples), ('Test', self.test_ds.samples)]:
                category_counts = {}
                for cat, obj, view in samples:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                
                print(f"   {split_name}:")
                for cat in sorted(self.categories):  # Sort for consistent order
                    count = category_counts.get(cat, 0)
                    pct = count / len(samples) * 100 if samples else 0
                    class_id = self.category_to_class[cat]
                    print(f"     {cat} (class {class_id}): {count:,} samples ({pct:.1f}%)")
        else:
            print(f"\n📈 Per-Model Statistics (showing top 10 models by sample count):")
            for split_name, samples in [('Train', self.train_ds.samples), ('Val', self.val_ds.samples), ('Test', self.test_ds.samples)]:
                model_counts = {}
                for cat, obj, view in samples:
                    model_id = f"{cat}/{obj}"
                    model_counts[model_id] = model_counts.get(model_id, 0) + 1
                
                # Sort by count and show top 10
                sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                print(f"   {split_name} (top 10 models):")
                for model_id, count in sorted_models:
                    pct = count / len(samples) * 100 if samples else 0
                    class_id = self.model_to_class.get(model_id, -1)
                    print(f"     {model_id} (class {class_id}): {count:,} samples ({pct:.1f}%)")
        
        # Check for overlap (should be none)
        train_val_overlap = train_models & val_models
        train_test_overlap = train_models & test_models
        val_test_overlap = val_models & test_models
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"   ⚠️  Model overlap detected!")
            if train_val_overlap:
                print(f"      Train/Val: {len(train_val_overlap)} models")
            if train_test_overlap:
                print(f"      Train/Test: {len(train_test_overlap)} models")
            if val_test_overlap:
                print(f"      Val/Test: {len(val_test_overlap)} models")
        else:
            print(f"   ✅ No model overlap between splits")

class RotationMNIST_DataLoader:
    """
    Builds train/val/test dataloaders for MNIST with rotations.
    """

    def __init__(self, root, batch_size, num_workers, opt,
                 rotation_range=(-90, 90), fixed_rotation=False,
                 val_split=0.1, download=True):

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.opt = opt
        self.rotation_range = rotation_range
        self.fixed_rotation = fixed_rotation
        self.val_split = val_split
        self.num_classes = 10

        # Base MNIST datasets
        full_train = MNIST(root=root, train=True, download=download)
        test = MNIST(root=root, train=False, download=download)

        # Split train into train + val
        val_size = int(len(full_train) * val_split)
        train_size = len(full_train) - val_size
        train_base, val_base = random_split(full_train, [train_size, val_size])

        # Transforms
        base_transform = T.Compose([T.Resize((28, 28)), T.ToTensor()])
        aug_transform = T.Compose([T.RandomResizedCrop(size=28, scale=(0.9, 1.0)), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), T.ToTensor()])

        # Use your existing TwoTransformNew class
        train_transform = TwoTransformNew(base_transform, aug_transform)
        eval_transform = TwoTransformNew(base_transform, base_transform)

        # Wrap with RotationMNISTDataset
        self.train_ds = RotationMNISTDataset(train_base, train_transform, opt,
                                             rotation_range, fixed_rotation, num_classes=self.num_classes)
        self.val_ds = RotationMNISTDataset(val_base, eval_transform, opt,
                                           rotation_range, fixed_rotation, num_classes=self.num_classes)
        self.test_ds = RotationMNISTDataset(test, eval_transform, opt,
                                            rotation_range, fixed_rotation, num_classes=self.num_classes)

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        print("✅ RotationMNIST DataLoaders built")
        print(f"   Train size: {len(self.train_ds)}")
        print(f"   Val size: {len(self.val_ds)}")
        print(f"   Test size: {len(self.test_ds)}")

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def get_datasets(self):
        return self.train_ds, self.val_ds, self.test_ds
