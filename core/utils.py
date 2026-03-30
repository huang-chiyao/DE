import os
import re
import math
import uuid
import random
import string
from itertools import combinations

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

from torch.nn.functional import cosine_similarity
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, kendalltau
import torch, numpy as np, pandas as pd, wandb, plotly.express as px
from typing import Iterable, List, Tuple, Union, Literal, Optional
import umap
import pandas as pd
import plotly.express as px
import wandb

from .model import *
from .loss import RnCLoss
from .dataloader import *
from tools.evaluation.angle import calculate_circular_mae, calculate_mae, calculate_circular_mae_split, viewpoint_error_3dof, viewpoint_error_6dof

from sklearn.metrics import (
    adjusted_mutual_info_score, 
    v_measure_score, 
    silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist, squareform
import warnings

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_transforms(split, aug):
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    return transform


def get_label_dim(dataset):
    if dataset in ['AgeDB']:
        label_dim = 1
    else:
        raise ValueError(dataset)
    return label_dim


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def save_regressor(regressor, reg_optimizers, opt, epoch, save_file):
    print('==> Saving regressor...')
    state = {
        'opt':           opt,
        'regressor':     regressor.state_dict(),
        'reg_optimizers': {
            f'optimizer_{i}': opt_i.state_dict()
            for i, opt_i in reg_optimizers.items()
        },
        'epoch':         epoch,
    }
    torch.save(state, save_file)
    del state


def set_optimizer(opt, models):
    """
    Initialize separate optimizers for each model.
    
    Args:
        opt: Parsed command-line options containing learning rate, momentum, etc.
        models: A dictionary of models for each DoF (e.g., {0: model_x, 1: model_y, ...}).

    Returns:
        A dictionary of optimizers, one for each DoF.
    """
    # Create a separate optimizer for each DoF model
    optimizers = {
        dof: torch.optim.SGD(
            models[dof].parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay
        )
        for dof in range(6)
    }
    return optimizers


def save_checkpoint(epoch, models, regressors, optimizers, best_errors, save_path):
    """Save all models, regressors, optimizers, and best errors to a checkpoint."""
    checkpoint = {'epoch': epoch}

    # Add all models, regressors, optimizers, and best errors for each DOF to the checkpoint
    for dof in range(6):
        checkpoint[f'model_{dof}'] = models[dof].state_dict()
        checkpoint[f'regressor_{dof}'] = regressors[dof].state_dict()
        checkpoint[f'optimizer_{dof}'] = optimizers[dof].state_dict()
        checkpoint[f'best_error_{dof}'] = best_errors[dof]

    torch.save(checkpoint, save_path)


def calculate_correlation(latents, labels):

    latent_norm = cdist(latents, latents, metric='euclidean').flatten()
    labels_norm = cdist(labels, labels, metric='cityblock').flatten()

    # Debug checks
    # print(f"latent_norm stats: min={latent_norm.min():.6f}, max={latent_norm.max():.6f}, "
    #       f"std={latent_norm.std():.6f}, has_nan={np.isnan(latent_norm).any()}")
    # print(f"labels_norm stats: min={labels_norm.min():.6f}, max={labels_norm.max():.6f}, "
    #       f"std={labels_norm.std():.6f}, has_nan={np.isnan(labels_norm).any()}")
    

    spearman_p, _ = spearmanr(latent_norm, labels_norm)
    kendall_p, _ = kendalltau(latent_norm, labels_norm)

    return spearman_p, kendall_p


def set_regressor(opt):
    """Build a set of regressors, each represented by a linear layer, based on the encoder type."""
    if opt.model == 'resnet18':
        output_dim = 512
    elif opt.model == 'resnet50':
        output_dim = 2048
    elif opt.model == 'ViT':
        output_dim = 768
    else:
        raise ValueError(f"Unsupported encoder type: {opt.model}")

    # Create a dictionary with six regressors indexed by numbers 0-5
    regressors = {dof: torch.nn.Linear(output_dim, 1).cuda() for dof in range(6)}
    return regressors


def set_loader(opt):

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    gaussian_transform = transforms.Compose([
        GaussianFlowAugmentation(mean=0.0, std=0.00),
    ])

    class TwoTransform:
        def __init__(self, transform1, transform2):
            self.transform1 = transform1
            self.transform2 = transform2

        def __call__(self, x):
            return [self.transform1(x), self.transform2(self.transform1(x))]
        
    transform = TwoTransform(transform, transform)

    # Build data loaders
    if opt.dataset in ['ShapeNetRenderingDataset']:
        
        loader = ShapeNetRendering_DataLoader(
            root=opt.data_folder,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            categories=opt.data_prefix,
            opt=opt
        )
        train_loader, valid_loader, test_loader = loader.get_loaders()
        
    elif opt.dataset in ['MPIIFaceGazeNormDataset']:
        loader = MPIIFaceGazeNorm_DataLoader(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            categories=opt.data_prefix,
            opt=opt
        )
        train_loader, valid_loader, test_loader = loader.get_loaders()

    elif opt.dataset in ['RotationMNISTDataset']:
        loader = RotationMNIST_DataLoader(
            root=opt.data_folder,
            opt=opt,
            rotation_range=(-90, 90),
            # train_ratio=0.8,
            val_split=0.1,
            # transform=transform,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
        )
        
        train_loader, valid_loader, test_loader = loader.get_loaders()
    else:
        raise ValueError(f"Your dataset and dataloader fucked up: {opt.data_folder}, {opt.dataset}")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(valid_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    return train_loader, valid_loader, test_loader

def calculate_intra_inter_class_distance_ratio(features, labels):
    """
    Calculate the ratio of intra-class to inter-class distances.
    Lower values indicate better clustering (tight intra-class, far inter-class).
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes < 2:
        return float('inf')  # Only one class, no inter-class distance
    
    # Calculate intra-class distances
    intra_distances = []
    for label in unique_labels:
        class_mask = labels == label
        class_features = features[class_mask]
        
        if len(class_features) > 1:
            # Calculate pairwise distances within the class
            distances = pdist(class_features)
            intra_distances.extend(distances)
    
    # Calculate inter-class distances (between class centroids)
    centroids = []
    for label in unique_labels:
        class_mask = labels == label
        class_features = features[class_mask]
        centroid = np.mean(class_features, axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    inter_distances = pdist(centroids)
    
    # Calculate ratio
    if len(intra_distances) == 0:
        return 0.0  # All classes have single samples
    
    mean_intra = np.mean(intra_distances)
    mean_inter = np.mean(inter_distances)
    
    if mean_inter == 0:
        return float('inf')
    
    return mean_intra / mean_inter

def calculate_classification_metrics(features, true_labels, n_clusters=None):
    """
    Calculate only the 4 requested classification metrics.
    
    Args:
        features: numpy array of shape (N, feature_dim)
        true_labels: numpy array of shape (N,) with true class labels
        n_clusters: number of clusters for KMeans (if None, uses number of unique labels)
    
    Returns:
        dict: Dictionary containing the 4 classification metrics
    """
    
    # Handle edge cases
    if len(features) == 0 or len(true_labels) == 0:
        return {
            'adjusted_mutual_info': float('nan'),
            'v_measure': float('nan'), 
            'silhouette_score': float('nan'),
            'intra_inter_ratio': float('nan')
        }
    
    # Encode labels to ensure they're integers starting from 0
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(true_labels.flatten())
    
    n_unique_labels = len(np.unique(encoded_labels))
    
    if n_clusters is None:
        n_clusters = n_unique_labels
    
    # Ensure we have enough samples and reasonable number of clusters
    n_samples = len(features)
    if n_samples < 2:
        return {
            'adjusted_mutual_info': float('nan'),
            'v_measure': float('nan'), 
            'silhouette_score': float('nan'),
            'intra_inter_ratio': float('nan')
        }
    
    n_clusters = min(n_clusters, n_samples - 1)
    n_clusters = max(n_clusters, 2)  # At least 2 clusters
    
    results = {}
    
    try:
        # Perform K-means clustering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(features)
        
        # Adjusted Mutual Information
        try:
            results['adjusted_mutual_info'] = adjusted_mutual_info_score(encoded_labels, predicted_labels)
        except:
            results['adjusted_mutual_info'] = float('nan')
        
        # V-Measure Score
        try:
            results['v_measure'] = v_measure_score(encoded_labels, predicted_labels)
        except:
            results['v_measure'] = float('nan')
        
        # Silhouette Score (using predicted clusters)
        try:
            if len(np.unique(predicted_labels)) > 1:
                results['silhouette_score'] = silhouette_score(features, predicted_labels)
            else:
                results['silhouette_score'] = float('nan')
        except:
            results['silhouette_score'] = float('nan')
        
        # Intra-Inter Class Distance Ratio (using true labels)
        try:
            results['intra_inter_ratio'] = calculate_intra_inter_class_distance_ratio(features, encoded_labels)
        except:
            results['intra_inter_ratio'] = float('nan')
    
    except Exception as e:
        print(f"Error in classification metrics calculation: {e}")
        # Return NaN for all metrics if something goes wrong
        results = {
            'adjusted_mutual_info': float('nan'),
            'v_measure': float('nan'), 
            'silhouette_score': float('nan'),
            'intra_inter_ratio': float('nan')
        }
    
    return results

def log_metrics_to_wandb(epoch, spearman_list=None, kendall_list=None, classification_metrics_list=None, 
                        regression_names=None, classification_names=None, test=False):
    """
    Log all metrics to wandb in a centralized function.
    
    Args:
        epoch: current epoch number
        spearman_list: list of spearman correlations for regression outputs
        kendall_list: list of kendall correlations for regression outputs  
        classification_metrics_list: list of classification metric dicts
        regression_names: list of regression output names
        classification_names: list of classification output names
    """
    prefix = 'test' if test else 'encoder'
    corr_log = {f"{prefix}/epoch": epoch}
    
    # Log regression metrics
    if spearman_list and kendall_list:
        for i, (s, k) in enumerate(zip(spearman_list, kendall_list)):
            if regression_names and i < len(regression_names):
                name_suffix = f"_{regression_names[i]}"
            else:
                name_suffix = f"_{i}"
            corr_log[f"{prefix}/spearman_corr{name_suffix}"] = s
            corr_log[f"{prefix}/kendall_corr{name_suffix}"] = k
    
    # Log classification metrics
    if classification_metrics_list and classification_names:
        for i, metrics in enumerate(classification_metrics_list):
            name_suffix = f"_{classification_names[i]}"
            corr_log[f"{prefix}/adj_mutual_info{name_suffix}"] = metrics['adjusted_mutual_info']
            corr_log[f"{prefix}/v_measure{name_suffix}"] = metrics['v_measure']
            corr_log[f"{prefix}/silhouette{name_suffix}"] = metrics['silhouette_score']
            corr_log[f"{prefix}/intra_inter_ratio{name_suffix}"] = metrics['intra_inter_ratio']
    
    # Log to wandb
    wandb.log(corr_log)
    
    return corr_log


def spearman_test(valid_loader, model, opt, device='cpu'):
    model.eval()
    device = torch.device(device)
    model.to(device)
    latent_list = []
    label_list = []
    
    with torch.no_grad():
        for idx, data_tuple in enumerate(valid_loader):
            image_0, labels = data_tuple
            bsz = labels.shape[0]
            # Select the dimension
            labels = labels.squeeze(0)
            # Image i and i + 1. No augmentation is needed here
            image_0 = image_0[0]
            image_0 = image_0.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Image i and i + 1 with augmentation
            f, _, _ = model(image_0)
            f = f.squeeze(1)
            # print(f.shape)
            # print(labels.shape)
            latent_list.append(f.cpu())
            if labels.ndim == 1:
                label_list.append(labels.unsqueeze(0).cpu())
            else:
                label_list.append(labels.cpu())
    
    # 1) stitch all the batches together
    latent_tensor = torch.cat(latent_list, dim=0)  # (N, D, feat_dim)
    label_tensor = torch.cat(label_list, dim=0)    # (N, D) or (N,) if D=1
    
    # 2) convert once to NumPy
    latent_array = latent_tensor.numpy()
    label_array = label_tensor.numpy()
    
    print(f"latent_array shape {latent_array.shape}")
    print(f"label_array shape {label_array.shape}")
    
    # Identify regression and classification outputs
    regression_indices = []
    regression_names = []
    classification_indices = []
    classification_names = []
    
    for i, output_type in enumerate(opt.output_list):
        # Check if this output is for classification
        is_classification = any(keyword in output_type.lower() for keyword in opt.classification_keywords)
        if is_classification:
            classification_indices.append(i)
            classification_names.append(output_type)
        else:
            regression_indices.append(i)
            regression_names.append(output_type)
    
    print(f"Found {len(regression_indices)} regression outputs: {regression_names}")
    print(f"Found {len(classification_indices)} classification outputs: {classification_names}")
    print(f"Regression indices: {regression_indices}")
    print(f"Classification indices: {classification_indices}")
    
    # Calculate regression metrics (existing logic)
    spearman_list = []
    kendall_list = []
    
    # Only calculate correlations for regression outputs
    for idx, i in enumerate(regression_indices):
        # Extract the i-th eigenvector across all samples.
        # Flatten the arrays to make them 1D
        if latent_array.ndim == 2:
            f_i = latent_array
        else:
            f_i = latent_array[:, i, :]
        
        if label_array.ndim == 1:
            label_i = label_array
        elif label_array.ndim == 2:
            label_i = label_array[:, i]
        else:
            label_i = label_array[:, i, :]
        
        print(f"Processing regression {regression_names[idx]} (dim {i}): latent shape {f_i.shape}, label shape {label_i.shape}")
        
        spearman_p, kendall_p = calculate_correlation(f_i, label_i)
        spearman_list.append(spearman_p)
        kendall_list.append(kendall_p)
    
    # Calculate classification metrics
    classification_metrics_list = []
    
    for idx, i in enumerate(classification_indices):
        # Extract features and labels for this classification task
        if latent_array.ndim == 2:
            # If latent_array is 2D, use all features
            f_i = latent_array
        else:
            # If latent_array is 3D, extract the i-th dimension
            f_i = latent_array[:, i, :]
        
        # Extract labels for this classification task
        if label_array.ndim == 1:
            label_i = label_array
        elif label_array.ndim == 2:
            label_i = label_array[:, i]
        else:
            label_i = label_array[:, i, :]
        
        # Flatten if needed
        if label_i.ndim > 1:
            label_i = label_i.flatten()
        
        print(f"Processing classification {classification_names[idx]} (dim {i}): latent shape {f_i.shape}, label shape {label_i.shape}")
        
        # Calculate classification metrics for this output
        metrics = calculate_classification_metrics(f_i, label_i)
        classification_metrics_list.append(metrics)
        
        # Print classification results
        print(f"Classification metrics for {classification_names[idx]}:")
        print(f"  Adjusted Mutual Info: {metrics['adjusted_mutual_info']:.4f}")
        print(f"  V-Measure Score: {metrics['v_measure']:.4f}")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Intra/Inter Ratio: {metrics['intra_inter_ratio']:.4f}")
    
    return spearman_list, kendall_list, classification_metrics_list, regression_names, classification_names


def spearman_test_for_baseline_rnc(valid_loader, model, opt=None, device='cpu'):
    model.eval()

    latent_list = []
    label_list = []

    with torch.no_grad():

        for idx, data_tuple in enumerate(valid_loader):

            image_0, labels = data_tuple

            # labels = labels.squeeze(0) 

            # Image i and i + 1. No augmentation is needed here
            image_0 = image_0[0]
            # image_0 = normalize_flow_tensor(image_0, max_magnitude=150.0)

            # if opt.dataset == "CFVDataset" and opt is not None and opt.output_dim == 1:
            #     sin_theta, cos_theta = labels[:,0], labels[:,1]
                
            #     # Compute the angle in radians
            #     labels = torch.atan2(sin_theta, cos_theta)
            # else:
            labels = labels.squeeze(0) 

            if torch.cuda.is_available():
                image_0 = image_0.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # Image i and i + 1 with augmentation
            f, _, _ = model(image_0)

            latent_list.append(f.to(device='cpu'))
            label_list.append(labels.to(device='cpu'))

    # 1) stitch all the batches together
    latent_tensor = torch.cat(latent_list, dim=0)  # (N, D, feat_dim)
    label_tensor  = torch.cat(label_list,  dim=0)  # (N, D) or (N,) if D=1

    # 2) convert once to NumPy
    latent_array = latent_tensor.numpy()
    label_array  = label_tensor.numpy()

    print(f"latent_array shape {latent_array.shape}")
    print(f"label_array shape {label_array.shape}")

    spearman_list = []
    kendall_list = []
    for i in range(opt.output_dim):
        # Extract the i-th eigenvector across all samples.
        f_i = latent_array
        # print(label_array.shape)
        label_i = label_array[:, i, :]

        spearman_p, kendall_p = calculate_correlation(f_i, label_i)
        spearman_list.append(spearman_p)
        kendall_list.append(kendall_p)

    return spearman_list, kendall_list

def plot_2d_pca_scatter_by_class(
    all_f_cls: np.ndarray,
    all_labels: np.ndarray,
    output_folder: str,
    epoch: int = None
):
    """
    1) Runs PCA(n_components=2) on all_f_cls (shape N×feat_dim).
    2) For each sample i, plots its 2D PCA coordinate (PC1, PC2) as a small dot,
       with its color determined by all_labels[i].  
    3) Saves the scatter under 
         "{output_folder}/pca2d_scatter_epoch_{epoch}.png"
       (or "pca2d_scatter.png" if epoch is None).
    Returns:
      • explained_var: array([v1, v2]) = PCA.explained_variance_ratio_
    """

    # Ensure the directory exists
    os.makedirs(output_folder, exist_ok=True)

    # 1) Fit 2-D PCA
    pca = PCA(n_components=2)
    pcs2 = pca.fit_transform(all_f_cls)            # shape: (N, 2)
    explained_var = pca.explained_variance_ratio_  # length=2

    # 2) Create scatter plot of all points, colored by class
    plt.figure(figsize=(6, 6))
    for cls_id in np.unique(all_labels):
        mask = (all_labels == cls_id)               # boolean mask for this class
        plt.scatter(
            pcs2[mask, 0], 
            pcs2[mask, 1],
            s=20,            # marker size; you can make this bigger/smaller
            alpha=0.3,       # 0.0 = fully transparent, 1.0 = opaque
            label=f"Class {cls_id}"
        )

    plt.xlabel(f"PC₁ (explained var = {explained_var[0]:.3f})")
    plt.ylabel(f"PC₂ (explained var = {explained_var[1]:.3f})")
    title = "2D PCA of f_cls (all samples colored by class)"
    if epoch is not None:
        title += f"  — epoch {epoch}"
    plt.title(title)

    plt.legend(loc="best", fontsize="small", framealpha=0.8)
    plt.tight_layout()

    # 3) Save the figure
    if epoch is not None:
        fname = f"pca2d_scatter_epoch_{epoch}.png"
    else:
        fname = "pca2d_scatter.png"
    save_path = os.path.join("/scratch/ngocbach/re/", output_folder, fname)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[PCA Scatter] Saved to: {save_path}")

    return explained_var

def spearman_test_with_class(valid_loader, model, opt, epoch=None):
    """
    Runs validation on `valid_loader`, returning:
      • spearman_list       (one ρ per output_dim)
      • kendall_list        (one τ per output_dim)
      • mean_cls_distance   (dict: class_id → average variance of f_cls)
    Additionally:
      - Prints validation accuracy & macro‐F1
      - Saves a 2D‐PCA circle‐plot of f_cls under opt.model_name/pca2d_epoch_{epoch}.png
    """

    model.eval()

    # ─── Buffers to collect everything ────────────────────────────────
    latent_list    = []  # store “f” for Spearman/Kendall
    label_list     = []  # store rotation labels for each batch
    cls_true_list  = []  # store ground-truth class label per sample
    cls_pred_list  = []  # store predicted class label per sample
    f_cls_list     = []  # store f_cls (feature before g_cls) per sample
    cls_label_list = []  # store cls_label again (for PCA coloring)

    # Build a map: class_id → [all f_cls vectors with that class]
    class_feature_map = {}

    with torch.no_grad():
        for idx, data_tuple in enumerate(valid_loader):
            image_0, rotation_labels, cls_labels = data_tuple

            # rotation_labels might be shape (1, R) or (1, R, D) → make it (R,) or (R, D)
            rotation_labels = rotation_labels.squeeze(0)
            bsz_rot = rotation_labels.shape[0]

            # image_0 is a tuple ([img_i], [img_{i+1}]) → take only the first view
            image_0 = image_0[0]  # shape: (1, C, H, W)

            # Move to GPU if available
            if torch.cuda.is_available():
                image_0         = image_0.cuda(non_blocking=True)
                rotation_labels = rotation_labels.cuda(non_blocking=True)
                cls_labels      = cls_labels.cuda(non_blocking=True)

            # ─── FORCE cls_labels to shape (batch_size_cls,) ───────────
            cls_labels = cls_labels.squeeze(0).view(-1)  # shape: (1,) in your setup
            bsz_cls   = cls_labels.shape[0]              # bsz_cls == 1

            # ─── Forward pass ─────────────────────────────────────────
            # f_proj, f, proj_coeffs, f_cls, cls_logits = model(image_0)
            f_proj, f, _, f_cls, cls_logits = model(image_0)
            f = f.squeeze(1)  # shape: (1, feat_dim)

            # 1) Collect classification predictions & truths
            preds = cls_logits.argmax(dim=1).cpu()    # shape: (1,)
            cls_pred_list.append(preds)
            cls_true_list.append(cls_labels.cpu())    # shape: (1,)

            # 2) Build class_feature_map[class_id] ← f_cls_cpu[i]
            f_cls_cpu = f_cls.cpu()  # shape: (1, feat_dim)
            for i in range(bsz_cls):  # range(1)
                cls_id = int(cls_labels[i].item())
                class_feature_map.setdefault(cls_id, []).append(f_cls_cpu[i])

            # 3) Keep every f_cls & cls_label for PCA later
            f_cls_list.append(f_cls_cpu)             # list of (1, feat_dim)
            cls_label_list.append(cls_labels.cpu())  # list of (1,)

            # 4) Collect rotation-features for Spearman/Kendall
            latent_list.append(f.cpu())              # shape: (1, feat_dim)
            label_list.append(rotation_labels.unsqueeze(0).cpu())
            #    rotation_labels.unsqueeze(0) has shape (1, R)
            #    Later we’ll concatenate these into (N, R) or (N,) if R=1

    # ─── STITCH EVERYTHING INTO BIG TENSORS/ARRAYS ────────────────────
    latent_tensor = torch.cat(latent_list, dim=0)  # shape: (N, feat_dim)
    label_tensor  = torch.cat(label_list,  dim=0)  # shape: (N, R) or (N,) if R=1

    # classification: true_labels & pred_labels as NumPy arrays
    true_labels = torch.cat(cls_true_list, dim=0).numpy()  # (N,)
    pred_labels = torch.cat(cls_pred_list, dim=0).numpy()  # (N,)

    # ─── Compute Spearman + Kendall ───────────────────────────────────
    latent_array = latent_tensor.numpy()   # shape: (N, feat_dim) or (N, output_dim, feat_dim)
    label_array  = label_tensor.numpy()    # shape: (N, ) or (N, D)

    spearman_list = []
    kendall_list  = []

    for i in range(opt.output_dim + 1 if opt.classification else opt.output_dim):
        if latent_array.ndim == 2:
            # If output_dim == 1, latent_array is already (N, feat_dim)
            f_i = latent_array            # shape: (N, feat_dim)
        else:
            # If output_dim > 1, latent_array is (N, output_dim, feat_dim)
            f_i = latent_array[:, i, :]   # shape: (N, feat_dim)

        # *** HERE IS THE FIX: labels must be 2-D for cdist ***
        #   label_array[:, i] is shape (N,), so we reshape to (N,1)
        label_i = label_array[:, i].reshape(-1, 1)  # now shape: (N, 1)

        spearman_p, kendall_p = calculate_correlation(f_i, label_i)
        spearman_list.append(spearman_p)
        kendall_list.append(kendall_p)

    # ─── Compute accuracy + macro‐F1 and print them ───────────────────
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score   (true_labels, pred_labels, average='macro')
    print(f"Validation Accuracy: {accuracy*100:>6.2f}%")
    print(f"Validation F1 (macro): {f1_macro:.4f}")

    # ─── Compute per-class “mean variance” of f_cls ──────────────────
    mean_cls_distance = {}
    for cls_id, f_cls_samples in class_feature_map.items():
        # f_cls_samples: list of Tensors, each (feat_dim,)
        stacked = torch.stack(f_cls_samples, dim=0)  # shape: (num_samples_of_class, feat_dim)
        var_vec = stacked.var(dim=0)                 # shape: (feat_dim,)
        # average variance across features
        mean_cls_distance[cls_id] = var_vec.mean().item()

    # ─── Build “all_f_cls” & “all_labels” for PCA ────────────────────
    all_f_cls  = torch.cat(f_cls_list,   dim=0).numpy()  # (N, feat_dim)
    all_labels = torch.cat(cls_label_list, dim=0).numpy() # (N,)

    # ─── 2D PCA + one-circle-per-class visualization ─────────────────
    explained_var = plot_2d_pca_scatter_by_class(
        all_f_cls     = all_f_cls,
        all_labels    = all_labels,
        output_folder = opt.model_name,
        epoch         = epoch
    )

    # Correct the variable names when printing:
    print(f"PCA explained variance (PC1, PC2): {explained_var[0]:.3f}, {explained_var[1]:.3f}")
    # for cid in sorted(class_centers.keys()):
    #     mu  = class_centers[cid]
    #     var = class_vars[cid]
    #     print(f"  Class {cid:>2d}:   mean=({mu[0]:.3f},{mu[1]:.3f})   var_sum={var:.4f}")

    return spearman_list, kendall_list, mean_cls_distance

#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import plotly.express as px
from itertools import combinations
from typing import Iterable, Optional, Union, Literal, List, Tuple
import torch
def plot_3d_latent(
    valid_loader: Iterable,
    model: torch.nn.Module,
    epoch: int,
    opt,
    device: Union[str, torch.device] = "cuda",
    pairs: Optional[Union[Literal["consecutive", "slide", 'permutation', 'all_pairs'],
                        List[Tuple[int, int, int]]]] = "permutation",
    max_points: Optional[int] = 10000,
    title_prefix: str = "v",
    outlier_percentile: float = 2.5,  # Remove top/bottom 2.5% as outliers
    log_to_wandb: bool = True,  # New parameter for wandb logging
):
    """
    3D PCA visualization matching the plot_latent structure but focused on 3D only.
    Enhanced with outlier removal and iterative ordering.
    Now supports wandb logging with static images.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import wandb
    
    # ----------------- collect latent ---------------------------------------
    model.eval()
    model.to(device := torch.device(device))
    latents, labels = [], []
    with torch.no_grad():
        for imgs, lbls in valid_loader:
            imgs = imgs[0].to(device, non_blocking=True) # (B,C,H,W)
            lbls = lbls.squeeze(0).to(device, non_blocking=True)
            _, f, _ = model(imgs) # f: (B,1,D)
            latents.append(f.squeeze(1).cpu())
            labels.append(lbls.cpu())
    
    X = torch.cat(latents, 0).numpy() # (N, D)
    Y = torch.cat(labels, 0).squeeze(-1).numpy()
    
    if max_points and X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], max_points, replace=False)
        X, Y = X[idx], Y[idx]
    
    # ----------------- projection -------------------------------------------
    # ----------------- choose eigenvectors ----------------------------------
    D = opt.output_dim
    if opt.method == "Domain":
        # --- pre-computed eigenvectors and running mean stored in the model ----
        V = model.prev_evecs[:, :D].detach().cpu().numpy() # (latent_dim, D)
        mu = model.running_mean.detach().cpu().numpy() # (latent_dim,)
    else:
        # --- compute eigenvectors directly from current batch of latents -------
        # 1. centre latents on their own mean
        mu = X.mean(axis=0, keepdims=False) # (latent_dim,)
        Xc_tmp = X - mu
        # 2. thin-SVD: Xc = U Σ Vᵀ → rows of Vᵀ are eigenvectors
        # V has shape (D, D) with descending singular values
        _, _, Vt = np.linalg.svd(Xc_tmp, full_matrices=False)
        Vt = Vt[:D, :]
        V = Vt.T # (latent_dim, D)
    
    # finally centre X once
    Xc = X - mu # (N, latent_dim)
    
    # ----------------- generate iterative triplets for 3D plotting ----------
    # Create iterative order: [v0-v1-v2] colored v0, [v1-v2-v3] colored v1, etc.
    eig_pairs = []
    for i in range(D):
        j = (i + 1) % D  # next dimension (wraps around)
        k = (i + 2) % D  # dimension after that (wraps around)
        eig_pairs.append((i, j, k))
    
    # ----------------- 3D plotting with outlier removal ---------------------
    for triplet_idx, (i, j, k) in enumerate(eig_pairs):
        print(f"3D PCA plot {triplet_idx+1}/{len(eig_pairs)}: {title_prefix}_{i} vs {title_prefix}_{j} vs {title_prefix}_{k}")
        coords = Xc @ V[:, [i, j, k]] # (N, 3)
        
        # -------- outlier removal -------------------------------------------
        # Calculate distances from the center for outlier detection
        center = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        
        # Remove outliers based on percentile
        lower_percentile = np.percentile(distances, outlier_percentile)
        upper_percentile = np.percentile(distances, 100 - outlier_percentile)
        
        # Keep points within the percentile range
        mask = (distances >= lower_percentile) & (distances <= upper_percentile)
        coords_filtered = coords[mask]
        Y_filtered = Y[mask] if Y.ndim == 1 else Y[mask]
        
        # -------- handle 1-D vs k-D labels ----------------------------------
        if Y.ndim == 1: # scalar labels → single plot
            label_names = ["label"]
            colours = [Y_filtered]
        else: # vector labels → one plot / column
            label_names = [f"label{i}"]
            colours = [Y_filtered[:, i]]
        
        # -------- emit one figure per label column --------------------------
        for cname, colour in zip(label_names, colours):
            # Create matplotlib 3D plot instead of plotly
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create scatter plot
            scatter = ax.scatter(
                coords_filtered[:, 0], 
                coords_filtered[:, 1], 
                coords_filtered[:, 2],
                c=colour.ravel(), 
                cmap='plasma', 
                alpha=0.6,
                s=20
            )
            
            # Set labels and title
            ax.set_xlabel(f"{title_prefix}_{i}")
            ax.set_ylabel(f"{title_prefix}_{j}")
            ax.set_zlabel(f"{title_prefix}_{k}")
            ax.set_title(f"{title_prefix}_{i}-{title_prefix}_{j}-{title_prefix}_{k} (colored by c_{i})")
            
            # Add colorbar
            # plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
            
            # Make axes equal (cube-like visualization)
            x_data, y_data, z_data = coords_filtered[:, 0], coords_filtered[:, 1], coords_filtered[:, 2]
            all_data = np.concatenate([x_data, y_data, z_data])
            global_min, global_max = np.min(all_data), np.max(all_data)
            padding = (global_max - global_min) * 0.1
            axis_min = global_min - padding
            axis_max = global_max + padding
            
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.set_zlim(axis_min, axis_max)
            
            # Set equal aspect ratio (as close as possible with matplotlib)
            ax.set_box_aspect([1,1,1])
            
            # Save to file
            filename = f"3d_pca_{epoch}_{title_prefix}_{i}-{title_prefix}_{j}-{title_prefix}_{k}_colored-{title_prefix}_{i}.png"
            output_path = os.path.join(opt.save_folder, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            # Log to wandb if enabled
            if log_to_wandb and wandb.run is not None:
                wandb.log({
                    f"3D_PCA_4/{title_prefix}_{i}-{title_prefix}_{j}-{title_prefix}_{k}": wandb.Image(
                        output_path, 
                        caption=f"Epoch {epoch}: {title_prefix}_{i}-{title_prefix}_{j}-{title_prefix}_{k} (outliers removed: {(~mask).sum()}/{len(mask)} points)"
                    )
                }, step=epoch)
            
            plt.close()  # Important: close the figure to free memory
            print(f"Saved 3D plot: {output_path} (outliers removed: {(~mask).sum()}/{len(mask)} points)")
            
            # Optional: Also create multiple viewing angles
            if log_to_wandb and wandb.run is not None:
                # Create additional views from different angles
                angles = [(30, 45), (0, 90), (90, 0)]  # (elevation, azimuth)
                angle_names = ['default', 'top', 'side']
                
                for angle, angle_name in zip(angles, angle_names):
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        coords_filtered[:, 0], 
                        coords_filtered[:, 1], 
                        coords_filtered[:, 2],
                        c=colour.ravel(), 
                        cmap='viridis', 
                        alpha=0.6,
                        s=15
                    )
                    
                    ax.set_xlabel(f"{title_prefix}_{i}")
                    ax.set_ylabel(f"{title_prefix}_{j}")
                    ax.set_zlabel(f"{title_prefix}_{k}")
                    ax.set_title(f"Epoch {epoch}: {angle_name} view")
                    
                    # Set viewing angle
                    ax.view_init(elev=angle[0], azim=angle[1])
                    
                    # Equal axes
                    ax.set_xlim(axis_min, axis_max)
                    ax.set_ylim(axis_min, axis_max)
                    ax.set_zlim(axis_min, axis_max)
                    ax.set_box_aspect([1,1,1])
                    
                    # Log this view to wandb
                    wandb.log({
                        f"3D_PCA_4_views/{title_prefix}_{i}-{title_prefix}_{j}-{title_prefix}_{k}_{angle_name}": wandb.Image(
                            fig, 
                            caption=f"Epoch {epoch}: {angle_name} view - {title_prefix}_{i}-{title_prefix}_{j}-{title_prefix}_{k}"
                        )
                    }, step=epoch)
                    
                    plt.close()
                    
def plot_latent(
    valid_loader: Iterable,
    model: torch.nn.Module,
    epoch: int,
    opt,
    device: Union[str, torch.device] = "cuda",
    pairs: Optional[Union[Literal["consecutive", "slide", 'permutation', 'all_pairs'],
                        List[Tuple[int, int]], List[Tuple[int, int, int]]]] = "permutation",
    max_points: Optional[int] = 10000,
    title_prefix: str = "PC",
    plot_3d: bool = False,  # New parameter to control 2D vs 3D plotting
):
    # ----------------- collect latent ---------------------------------------
    model.eval()
    model.to(device := torch.device(device))
    latents, labels = [], []
    with torch.no_grad():
        for imgs, lbls in valid_loader:
            imgs = imgs[0].to(device, non_blocking=True) # (B,C,H,W)
            lbls = lbls.squeeze(0).to(device, non_blocking=True)
            _, f, _ = model(imgs) # f: (B,1,D)
            latents.append(f.squeeze(1).cpu())
            labels.append(lbls.cpu())
    X = torch.cat(latents, 0).numpy() # (N, D)
    Y = torch.cat(labels, 0).squeeze(-1).numpy()
    if max_points and X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], max_points, replace=False)
        X, Y = X[idx], Y[idx]
    
    # ----------------- projection -------------------------------------------
    # ----------------- choose eigenvectors ----------------------------------
    D = opt.output_dim
    if opt.method == "Domain":
        # --- pre-computed eigenvectors and running mean stored in the model ----
        V = model.prev_evecs[:, :D].detach().cpu().numpy() # (latent_dim, D)
        mu = model.running_mean.detach().cpu().numpy() # (latent_dim,)
    else:
        # --- compute eigenvectors directly from current batch of latents -------
        # 1. centre latents on their own mean
        mu = X.mean(axis=0, keepdims=False) # (latent_dim,)
        Xc_tmp = X - mu
        # 2. thin-SVD: Xc = U Σ Vᵀ → rows of Vᵀ are eigenvectors
        # V has shape (D, D) with descending singular values
        _, _, Vt = np.linalg.svd(Xc_tmp, full_matrices=False)
        Vt = Vt[:D, :]
        V = Vt.T # (latent_dim, D)
    
    # finally centre X once
    Xc = X - mu # (N, D)
    
    # ----------------- generate pairs/triplets ------------------------------
    if plot_3d:
        # Generate triplets for 3D plotting
        if pairs is None or pairs == "consecutive":
            eig_pairs = [(i, i + 1, i + 2) for i in range(0, D - 2, 3)]
        elif pairs == "slide":
            eig_pairs = [(i, i + 1, i + 2) for i in range(D - 2)]
        elif pairs == "permutation" or pairs == "all_pairs":
            # Generate all possible triplets (combinations) of indices
            eig_pairs = list(combinations(range(D), 3))
        else:
            eig_pairs = pairs
    else:
        # Generate pairs for 2D plotting (original behavior)
        if pairs is None or pairs == "consecutive":
            eig_pairs = [(i, i + 1) for i in range(0, D - 1, 2)]
        elif pairs == "slide":
            eig_pairs = [(i, i + 1) for i in range(D - 1)]
        elif pairs == "permutation" or pairs == "all_pairs":
            # Generate all possible pairs (combinations) of indices
            eig_pairs = list(combinations(range(D), 2))
        else:
            eig_pairs = pairs
    
    # ----------------- plotting ---------------------------------------------
    if plot_3d:
        # 3D plotting
        for i, j, k in eig_pairs:
            coords = Xc @ V[:, [i, j, k]] # (N, 3)
            
            # -------- handle 1-D vs k-D labels ----------------------------------
            if Y.ndim == 1: # scalar labels → single plot
                label_names = ["label"]
                colour_cols = [Y]
            else: # vector labels → one plot / column
                label_names = [f"label{l}" for l in range(Y.shape[1])]
                colour_cols = [Y[:, l] for l in range(Y.shape[1])]
            
            # -------- emit one figure per label column --------------------------
            for cname, colour in zip(label_names, colour_cols):
                df = pd.DataFrame({
                    f"{title_prefix}{i}": coords[:, 0].ravel(),
                    f"{title_prefix}{j}": coords[:, 1].ravel(),
                    f"{title_prefix}{k}": coords[:, 2].ravel(),
                    cname: colour.ravel(),
                })
                
                fig = px.scatter_3d(
                    df,
                    x=f"{title_prefix}{i}",
                    y=f"{title_prefix}{j}",
                    z=f"{title_prefix}{k}",
                    color=cname,
                    title=f"Epoch {epoch}: {title_prefix}{i} vs {title_prefix}{j} vs {title_prefix}{k} — {cname}",
                )
                
                wandb.log({
                    f"encoder_visualization/PC{i}-PC{j}-PC{k} ({cname})": fig,
                    "encoder_visualization/epoch": epoch
                })
    else:
        # 2D plotting (original behavior)
        for i, j in eig_pairs:
            coords = Xc @ V[:, [i, j]] # (N, 2)
            
            # -------- handle 1-D vs k-D labels ----------------------------------
            if Y.ndim == 1: # scalar labels → single plot
                label_names = ["label"]
                colour_cols = [Y]
            else: # vector labels → one plot / column
                label_names = [f"label{l}" for l in range(Y.shape[1])]
                colour_cols = [Y[:, l] for l in range(Y.shape[1])]
            
            # -------- emit one figure per label column --------------------------
            for cname, colour in zip(label_names, colour_cols):
                df = pd.DataFrame({
                    f"{title_prefix}{i}": coords[:, 0].ravel(),
                    f"{title_prefix}{j}": coords[:, 1].ravel(),
                    cname: colour.ravel(),
                })
                
                fig = px.scatter(
                    df,
                    x=f"{title_prefix}{i}",
                    y=f"{title_prefix}{j}",
                    color=cname,
                    title=f"Epoch {epoch}: {title_prefix}{i} vs {title_prefix}{j} — {cname}",
                    width=1200,  # Set larger width
                    height=800,  # Set larger height
                )
                # fig.update_traces(marker=dict(size=0.2))
                
                wandb.log({
                    f"encoder_visualization/PC{i}-PC{j} ({cname})": fig,
                    "encoder_visualization/epoch": epoch
                })
                
def plot_latent_classification(
    valid_loader: Iterable,
    model: torch.nn.Module,
    epoch: int,
    opt,
    device: Union[str, torch.device] = "cuda",
    max_points: Optional[int] = 10000,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
):
    """
    Plot classification latent space using UMAP dimensionality reduction
    Args:
        valid_loader: DataLoader for validation data
        model: The model with classification capabilities
        epoch: Current epoch number
        opt: Options object containing model configuration
        device: Device to run inference on
        max_points: Maximum number of points to plot (for performance)
        umap_n_neighbors: UMAP n_neighbors parameter
        umap_min_dist: UMAP min_dist parameter
        random_state: Random state for reproducibility
    """
    # ----------------- collect classification features ------------------
    model.eval()
    model.to(device := torch.device(device))
    classification_features = []
    labels = []
    
    with torch.no_grad():
        for imgs, lbls in valid_loader:
            imgs = imgs[0].to(device, non_blocking=True)  # (B,C,H,W)
            lbls = lbls.squeeze(0).to(device, non_blocking=True)
            
            # Get model outputs
            f_proj, f, proj_coeffs = model(imgs)  # f_proj: (B, output_dim + 1, feature_dim)
            
            # Extract classification features (last slice of f_proj)
            if opt.method == "Domain":
                # For Domain method, we take the last feature vector
                cls_features = f_proj[:, -1, :].cpu()  # (B, feature_dim)
            else:
                # For other methods, we take the last feature vector
                cls_features = f_proj.cpu()
            classification_features.append(cls_features)
            labels.append(lbls[:, -1].cpu())
    
    # Concatenate all features and labels
    X = torch.cat(classification_features, 0).numpy()  # (N, feature_dim)
    Y = torch.cat(labels, 0).numpy()  # (N,) or (N, num_classes)
    
    # Handle multi-class labels if needed
    if Y.ndim > 1:
        # If labels are one-hot encoded, convert to class indices
        if Y.shape[1] > 1:
            Y = np.argmax(Y, axis=1)
        else:
            Y = Y.squeeze()
    
    # Subsample if too many points
    if max_points and X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], max_points, replace=False)
        X, Y = X[idx], Y[idx]
    
    # ----------------- UMAP dimensionality reduction -------------------
    n_components = 1 if opt.classification_dim == 1 else 2
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state
    )
    embedding = reducer.fit_transform(X)  # (N, n_components)
    
    # ----------------- Color mapping setup (blue to yellow based on class) ---------
    # Create custom color scale from dark blue to yellow
    # This maps class labels to colors: lowest class = dark blue, highest class = yellow
    min_class = int(np.min(Y))
    max_class = int(np.max(Y))
    
    # Custom color scale from dark blue to yellow
    color_scale = [
        [0.0, '#000080'],    # Dark blue
        [0.2, '#0040FF'],    # Blue
        [0.4, '#0080FF'],    # Light blue
        [0.6, '#40FFFF'],    # Cyan
        [0.8, '#80FF80'],    # Light green
        [1.0, '#FFFF00']     # Yellow
    ]
    
    # ----------------- plotting -----------------------------------------
    if opt.classification_dim == 1:
        # 1D plot: show features in a line with color gradient based on class labels
        # Create y-coordinates with slight jitter for better visibility
        np.random.seed(random_state)
        y_coords = np.random.normal(0, 0.05, len(embedding))
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'y_jitter': y_coords,
            'class': Y.astype(int),  # Keep as integer for proper color mapping
            'class_str': Y.astype(str)  # String version for hover
        })
        
        # Create 1D scatter plot with color gradient based on class
        fig = px.scatter(
            df,
            x='UMAP1',
            y='y_jitter',
            color='class',
            hover_data=['class_str'],
            title=f"Epoch {epoch}: UMAP Classification Latent Space (1D)",
            labels={
                'UMAP1': 'UMAP Dimension 1',
                'y_jitter': 'Jitter (for visualization)',
                'class': 'Class Label',
                'class_str': 'Class'
            },
            color_continuous_scale=color_scale,
            range_color=[min_class, max_class]
        )
        
        # Update layout for 1D visualization
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        fig.update_layout(
            width=800,
            height=400,
            yaxis=dict(
                title="",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.3, 0.3]
            ),
            showlegend=False,
            coloraxis_colorbar=dict(
                title=f"Class Label<br>({min_class}→{max_class})<br>Blue→Yellow",
                titleside="right",
                tickmode="linear",
                tick0=min_class,
                dtick=1 if (max_class - min_class) <= 10 else (max_class - min_class) // 5
            )
        )
    
    else:
        # 2D plot with color gradient based on class labels
        df = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'class': Y.astype(int),  # Keep as integer for proper color mapping
            'class_str': Y.astype(str)  # String version for hover
        })
        
        # Create 2D scatter plot with color gradient based on class
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            color='class',
            hover_data=['class_str'],
            title=f"Epoch {epoch}: UMAP Classification Latent Space (2D)",
            labels={
                'UMAP1': 'UMAP Dimension 1',
                'UMAP2': 'UMAP Dimension 2',
                'class': 'Class Label',
                'class_str': 'Class'
            },
            color_continuous_scale=color_scale,
            range_color=[min_class, max_class]
        )
        
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.update_layout(
            width=800,
            height=600,
            showlegend=False,
            coloraxis_colorbar=dict(
                title=f"Class Label<br>({min_class}→{max_class})<br>Blue→Yellow",
                titleside="right",
                tickmode="linear",
                tick0=min_class,
                dtick=1 if (max_class - min_class) <= 10 else (max_class - min_class) // 5
            )
        )
    
    # Log to wandb
    wandb.log({
        "encoder_visualization/umap_latent_space": fig,
        "encoder_visualization/epoch": epoch,
    })

def plot_pca(X, y, save_path, label_dim=None, dim=2, show=False):
    """
    Minimal PCA scatter saver.

    Parameters
    ----------
    X : (N, D) array-like
        Feature matrix.
    y : (N,) or (N, K) array-like
        Labels. If multi-dim, pick column via `label_dim` (default=0).
    save_path : str or Path
        File to write (extension sets format: .png, .pdf, .svg, ...).
    label_dim : int, optional
        Which label column to color by when y.ndim > 1. Default=0.
    dim : {2, 3}
        Number of principal components to plot.
    show : bool
        If True, also display the figure.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Handle multi-dimensional labels
    if y.ndim > 1:
        if label_dim is None:
            label_dim = 0
        y = y[:, label_dim]

    # Center data & compute PCA via SVD (avoids scikit-learn dependency)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:dim]                           # (dim, D)
    Xp = Xc @ comps.T                          # (N, dim)
    var_ratio = (S[:dim] ** 2).sum() / (S ** 2).sum()

    # Decide discrete vs continuous coloring
    y_unique = np.unique(y)
    discrete = (y.dtype.kind in "iu" and len(y_unique) <= 20) or len(y_unique) <= 10

    fig = plt.figure(figsize=(6, 6))
    if dim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection="3d")
        if discrete:
            cmap = plt.cm.get_cmap("tab10", len(y_unique))
            color_map = {lab: cmap(i) for i, lab in enumerate(y_unique)}
            colors = [color_map[v] for v in y]
        else:
            norm = plt.Normalize(y.min(), y.max())
            colors = plt.cm.viridis(norm(y))
        ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], c=colors, s=10, alpha=0.9)
        ax.set_zlabel("PC3")
    else:
        ax = fig.add_subplot(111)
        if discrete:
            cmap = plt.cm.get_cmap("tab10", len(y_unique))
            color_map = {lab: cmap(i) for i, lab in enumerate(y_unique)}
            colors = [color_map[v] for v in y]
        else:
            norm = plt.Normalize(y.min(), y.max())
            colors = plt.cm.viridis(norm(y))
        ax.scatter(Xp[:, 0], Xp[:, 1], c=colors, s=10, alpha=0.9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA ({100*var_ratio:.1f}% var)")

    # Legend or colorbar
    if discrete:
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                       markerfacecolor=color_map[lab], label=str(lab))
            for lab in y_unique
        ]
        ax.legend(handles=handles, frameon=False, title="Label", loc="best")
    else:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        fig.colorbar(mappable, ax=ax, shrink=0.8, label="Label")

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

import torch
import psutil
import resource
import time
import wandb

def measure_train_memory(train_fn, *args, **kwargs):
    """Run a training function and measure peak CPU/GPU memory usage and time."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    process = psutil.Process()
    cpu_before = process.memory_info().rss / (1024 ** 3)
    start_time = time.time()

    result = train_fn(*args, **kwargs)

    torch.cuda.synchronize()
    end_time = time.time()

    # GPU peak memory
    gpu_peak = None
    if torch.cuda.is_available():
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # CPU peak memory (Linux/macOS)
    peak_cpu_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # in GB

    print(f"\n🔹 Training duration: {end_time - start_time:.2f} sec")
    if gpu_peak is not None:
        print(f"🧠 Peak GPU memory: {gpu_peak:.2f} GB")
    print(f"💻 Peak CPU memory (max RSS): {peak_cpu_mem:.2f} GB")

    wandb.log({
        "encoder/epoch": kwargs.get("epoch", 0),
        "encoder/peak_gpu_mem_GB": gpu_peak,
        "encoder/peak_cpu_mem_GB": peak_cpu_mem,
        "encoder/train_duration_sec": end_time - start_time
    })

    return result
import torch

import torch

def extract_prev_infer_coeffs(train_loader, model, opt, device="cuda"):
    """
    Extract previous inference coefficients as a single tensor (N, C_total),
    sorted by dataset order using provided indices.
    Assumes model returns `coeffs` as a tensor of shape (B, C_total)
    and that each batch is (image_batch, labels_batch, indices).
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    all_coeffs = []
    all_indices = []

    with torch.no_grad():
        for _, (image_batch, labels_batch, indices) in enumerate(train_loader):
            inputs = image_batch[0].to(device, non_blocking=True)
            indices = indices.cpu()

            if opt.method == "Domain":
                f_proj, f, coeffs = model(inputs)
                all_coeffs.append(coeffs.cpu())
                all_indices.append(indices)

    # Concatenate all batches
    coeff_tensor = torch.cat(all_coeffs, dim=0)   # (N, C_total)
    indices_tensor = torch.cat(all_indices, dim=0)

    # Sort by dataset order (important if loader was shuffled)
    sort_order = torch.argsort(indices_tensor)
    coeff_tensor = coeff_tensor[sort_order][:, :len(opt.prev_output_list)]

    print(f"[extract_prev_infer_coeffs] Final coeff tensor shape: {coeff_tensor.shape}")
    return coeff_tensor
