import argparse
import os
import sys
import logging
import torch
import torch.nn.functional as F
import time
from core.dataset import *
from core.utils import *
from core.loss import RnCLoss
from core.dataloader import *
import wandb

from tools.evaluation.angle import viewpoint_error_2dof, viewpoint_error_4dof

def test(test_loader, model, regressor, opt, device="cpu", val=False, use_cls=False, epoch=None):
    model.eval()
    regressor.eval()
    
    # (1) Make sure model + regressor are on the correct device
    device = torch.device(device)
    model.to(device)
    regressor.to(device)
    
    all_outputs = []
    all_labels = []
    all_features = []
    all_coefficients = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if not use_cls:
                image_batch, labels_batch = batch
            else:
                image_batch, labels_batch, _ = batch
                
            # Move inputs and labels onto the chosen device
            inputs = image_batch[0].to(device, non_blocking=True)
            labels = labels_batch.to(device, non_blocking=True)
            # Forward pass through the encoder + regressor
            if opt.method == "Domain":
                if not use_cls:
                    features, f, coeffs = model(inputs)
                else:
                    features, f, coeffs, _ = model(inputs) 
                processed = regressor(f)
                all_features.append(features.cpu())
                all_coefficients.append(coeffs.cpu())
            elif opt.method == "RNC":
                features, _, _ = model(inputs)
                processed = regressor(features)
            elif opt.method == "L1":
                processed = regressor(model(inputs))
            else:
                raise ValueError(f"Your method is not supported: {opt.method}")
                
            # Move everything back to CPU before storing
            all_outputs.append(processed.cpu())
            all_labels.append(labels.cpu())
    
    # (2) Concatenate across batches
    outputs = torch.cat(all_outputs, dim=0)  # shape (N, 2) or (N, output_dim)
    labels = torch.cat(all_labels, dim=0)    # same shape
    
    print(f"Shapes — outputs: {outputs.shape}, labels: {labels.shape}")
    
    # Determine logging prefix and step metric based on validation or test
    prefix = "val" if val else "test"
    get_metric(prefix, outputs, labels, opt, epoch=epoch)

                       
def get_metric(prefix, outputs, labels, opt, epoch=None):
    step_metric = f"{prefix}/epoch"
    # Use epoch if provided, otherwise use step 1
    step_value = epoch if epoch is not None else 1
    
    # Handle classification if enabled
    if hasattr(opt, 'classification') and opt.classification:
        # Split outputs and labels into regression and classification parts
        regression_outputs = outputs[:, :-1]  # All but last dimension
        classification_outputs = outputs[:, -1]  # Last dimension
        regression_labels = labels[:, :-1, :]
        classification_labels = labels[:, -1]
        
        # Evaluate classification
        # Round predictions to nearest integer and clamp to valid range
        classification_predictions = torch.round(classification_outputs)
        # Assuming class labels start from 0, find the max class from labels
        max_class = int(classification_labels.max().item())
        min_class = int(classification_labels.min().item())
        classification_predictions = torch.clamp(classification_predictions, min=min_class, max=max_class)
        
        # Calculate classification accuracy
        # print(min_class)
        # print(max_class)
        # print(classification_predictions[0, :5, :])
        # print(classification_labels[0, :5, :])
        correct_predictions = (classification_predictions == classification_labels).float()
        classification_accuracy = correct_predictions.mean().item()
        
        # Calculate classification MAE (for continuous prediction quality)
        classification_mae = torch.abs(classification_outputs - classification_labels).mean().item()
        
        # Log classification metrics
        wandb.log({
            step_metric: step_value,
            f"{prefix}/classification_accuracy": classification_accuracy,
            f"{prefix}/classification_mae": classification_mae,
            f"{prefix}/num_classes": max_class - min_class + 1,
        })
        
        print(f"Classification Accuracy: {classification_accuracy:.4f}, Classification MAE: {classification_mae:.4f}")
        
        # Use regression parts for the rest of the evaluation
        outputs = regression_outputs
        labels = regression_labels
        
        print(outputs.shape)
        print(labels.shape)
    
    # (3) Compute metrics based on output dimension
    if opt.dataset in ['KITTI_OpticalFlow_Dataset', 'TartanAir_OpticalFlow_Dataset', 'EuRoC_OpticalFlow_Dataset']:
        # Optical flow datasets: outputs and labels are in pixel units
        if opt.output_dim == 4:
            # Current use is dx, dz, droll, dpitch
            mae_translation = calculate_mae(outputs[:,:2], labels[:,:2])
            mae_rotation = calculate_circular_mae(outputs[:,2:], labels[:,2:])
            std = torch.std(outputs - labels, dim=0)
            
            # Log optical flow metrics (4D) with proper step metric
            wandb.log({
                step_metric: step_value,
                f"{prefix}/mae_x": mae_translation[0],
                f"{prefix}/mae_z": mae_translation[1],
                f"{prefix}/mae_roll": mae_rotation[0],
                f"{prefix}/mae_pitch": mae_rotation[1],
                f"{prefix}/std_dx": std[0],
                f"{prefix}/std_dz": std[1], 
                f"{prefix}/std_droll": std[2],
                f"{prefix}/std_dpitch": std[3],
            })
            
            # Create error distribution plots
            errors = outputs - labels
            error_data = []
            for i, component in enumerate(['dx', 'dz', 'droll', 'dpitch']):
                for error in errors[:, i]:
                    error_data.append([component, error.item()])
            
            error_table = wandb.Table(data=error_data, columns=["Component", "Error"])
            if 'val' not in prefix:
                wandb.log({
                    step_metric: step_value,
                    f"{prefix}/error_distribution": wandb.plot.histogram(error_table, "Error", 
                                                                        title=f"{prefix.capitalize()} Error Distribution")
                })
            
            print(f"MAE Translation: {mae_translation:.4f}, MAE Rotation: {mae_rotation:.4f}")
            
        elif opt.output_dim == 6:
            # Current use is dx, dz, dsinroll, dcosroll, dsinpitch, dcospitch
            mae_translation = calculate_mae(outputs[:,:2], labels[:,:2])
            mae_rotation_roll = calculate_circular_mae_split(
                outputs[:,2],  # Fixed: changed index from 3 to 2
                outputs[:,3],  # Fixed: changed index from 4 to 3
                labels[:,2],   # Fixed: changed index from 3 to 2
                labels[:,3]    # Fixed: changed index from 4 to 3
            )
            mae_rotation_pitch = calculate_circular_mae_split(
                outputs[:,4],  # Fixed: changed index from 5 to 4
                outputs[:,5],  # Fixed: changed index from 6 to 5
                labels[:,4],   # Fixed: changed index from 5 to 4
                labels[:,5]    # Fixed: changed index from 6 to 5
            )
            std = torch.std(outputs - labels, dim=0)
            
            # Log optical flow metrics (6D) with proper step metric
            wandb.log({
                step_metric: step_value,
                f"{prefix}/mae_x": mae_translation[0],
                f"{prefix}/mae_z": mae_translation[1],
                f"{prefix}/mae_roll": mae_rotation_roll,
                f"{prefix}/mae_pitch": mae_rotation_pitch,
                f"{prefix}/std_dx": std[0],
                f"{prefix}/std_dz": std[1],
                f"{prefix}/std_dsinroll": std[2],
                f"{prefix}/std_dcosroll": std[3],
                f"{prefix}/std_dsinpitch": std[4],
                f"{prefix}/std_dcospitch": std[5],
                f"{prefix}/output_dim": opt.output_dim
            })
            
            # Create detailed error analysis
            errors = outputs - labels
            error_data = []
            components = ['dx', 'dz', 'dsinroll', 'dcosroll', 'dsinpitch', 'dcospitch']
            for i, component in enumerate(components):
                for error in errors[:, i]:
                    error_data.append([component, error.item()])
            
            error_table = wandb.Table(data=error_data, columns=["Component", "Error"])
            if 'val' not in prefix:
                wandb.log({
                    step_metric: step_value,
                    f"{prefix}/error_distribution": wandb.plot.histogram(error_table, "Error",
                                                                        title=f"{prefix.capitalize()} Error Distribution")
                })
            
            print(f"MAE Translation: {mae_translation:.4f}, MAE Roll: {mae_rotation_roll:.4f}, MAE Pitch: {mae_rotation_pitch:.4f}")
            
    elif opt.dataset in ['ShapeNetRenderingDataset']:
        # The output depend on the dimension. 3 mean az, el, ro; 6 means sin_az, cos_az, sin_el, cos_el, sin_ro, cos_ro, 
        num_dimension_display = opt.output_dim
        if opt.output_dim == 3 and "az_el_ip_no_split" == opt.output_type:
            mae_error = calculate_circular_mae(outputs, labels.squeeze())
            errors = viewpoint_error_3dof(outputs, labels.squeeze())
        elif opt.output_dim == 6 and "az_el_ip" == opt.output_type:
            mae_error = calculate_mae(outputs, labels.squeeze())
            errors = viewpoint_error_6dof(outputs, labels.squeeze())
        elif opt.output_dim == 2 and "az_el_no_split" == opt.output_type:
            mae_error = calculate_circular_mae(outputs, labels.squeeze())
            errors = viewpoint_error_2dof(outputs, labels.squeeze())
        elif opt.output_dim == 4 and "az_el_az_el_no_split" == opt.output_type:
            mae_error = calculate_circular_mae(outputs[:,:2], labels[:,:2].squeeze())
            errors = viewpoint_error_2dof(outputs[:,:2], labels[:,:2].squeeze())
            num_dimension_display = 2
        elif opt.output_dim == 6 and "az_el_az_el_az_el_no_split" == opt.output_type:
            mae_error = calculate_circular_mae(outputs[:,:2], labels[:,:2].squeeze())
            errors = viewpoint_error_2dof(outputs[:,:2], labels[:,:2].squeeze())
            num_dimension_display = 2
        elif opt.output_dim == 4 and "az_el" == opt.output_type:
            mae_error = calculate_mae(outputs, labels.squeeze())
            errors = viewpoint_error_4dof(outputs, labels.squeeze())
        else:
            raise ValueError(f"Output dim for ShapeNetRenderingDataset and output_type has to match, got {opt.output_dim} and {opt.output_type}")
        median_err = errors.median().item()
        threshold_30 = 30.0 * torch.pi / 180.0 # 30 degrees to radians
        threshold_15 = 15.0 * torch.pi / 180.0 # 15 degrees to radians
        acc30 = (errors < threshold_30).float().mean().item()
        acc15 = (errors < threshold_15).float().mean().item()
        
        # Log ShapeNet metrics (3D) with proper step metric
        wandb.log({
            step_metric: step_value,
            f"{prefix}/median_error_deg": median_err * 180 / torch.pi,
            f"{prefix}/accuracy_30deg": acc30,
            f"{prefix}/accuracy_15deg": acc15,
            **{f"{prefix}/mae_dim{i}": mae_error[i].item() for i in range(num_dimension_display)}
        })
        
        # Create error distribution plot
        errors_deg = errors * 180 / torch.pi
        error_data = [[err.item()] for err in errors_deg]
        error_table = wandb.Table(data=error_data, columns=["Error_Degrees"])
        if 'val' not in prefix:
            wandb.log({
                step_metric: step_value,
                f"{prefix}/viewpoint_error_distribution": wandb.plot.histogram(error_table, "Error_Degrees",
                                                                            title=f"{prefix.capitalize()} Viewpoint Error Distribution")
            })
        mae_str = ", ".join([f"MAE_dim{i}: {mae_error[i]:.4f}" for i in range(len(mae_error))])
        print(f"Median Error: {median_err * 180 / torch.pi:.2f}°, Acc@30°: {acc30:.4f}, {mae_str}")
    
    else:
        raise ValueError(f"The evaluation of your dataset is not supported: {opt.dataset}")


def test_retrieval(test_loader_retrieval, model, regressor, opt, device="cuda", use_cls=False):
    """
    Retrieval test function that uses eigenvectors from model.prev_evecs
    model.prev_evecs: tensor of shape (feature_dim, total_eigen_dim) where each column is an eigenvector
    """
    model.eval()
    regressor.eval()
    
    # Make sure model + regressor are on the correct device
    device = torch.device(device)
    model.to(device)
    regressor.to(device)
    
    # Get eigenvectors from model
    eigenvectors = model.prev_evecs  # Shape: (feature_dim, total_eigen_dim)
    
    all_features1 = []
    all_features2 = []
    all_processed_features1 = []
    all_processed_features2 = []
    all_cam_params1 = []
    all_cam_params2 = []
    all_distances = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader_retrieval):
            # Correct unpacking based on your description
            image1, image2, cam_param1, cam_param2 = batch
            
            # Move inputs to device
            img1 = image1.to(device, non_blocking=True)
            img2 = image2.to(device, non_blocking=True)
            cam1 = cam_param1.to(device, non_blocking=True)
            cam2 = cam_param2.to(device, non_blocking=True)
            
            # Forward pass for image1
            if opt.method == "Domain":
                if not use_cls:
                    features1, f1, coeffs1 = model(img1)
                else:
                    features1, f1, coeffs1, _ = model(img1)
                processed1 = regressor(f1)
            elif opt.method == "RNC":
                features1, _, _ = model(img1)
                processed1 = regressor(features1)
            elif opt.method == "L1":
                features1 = model(img1)
                processed1 = regressor(features1)
            else:
                raise ValueError(f"Your method is not supported: {opt.method}")
            
            # Forward pass for image2
            if opt.method == "Domain":
                if not use_cls:
                    features2, f2, coeffs2 = model(img2)
                else:
                    features2, f2, coeffs2, _ = model(img2)
                processed2 = regressor(f2)
            elif opt.method == "RNC":
                features2, *_ = model(img2)
                processed2 = regressor(features2)
            elif opt.method == "L1":
                features2 = model(img2)
                processed2 = regressor(features2)
            else:
                raise ValueError(f"Your method is not supported: {opt.method}")
            
            # Apply high-dimensional operator using eigenvectors
            # Calculate parameter differences
            param_diff = cam2 - cam1  # Shape: (batch_size, num_params)
            
            # Apply eigenvector transformation to f1 (before regressor)
            if opt.method == "Domain":
                base_features = f1  # Use the intermediate features
            else:
                base_features = features1  # Use raw features for other methods
            
            # Apply the operator: f1_transformed = f1 + sum(eigenvector_i * param_diff_i)
            # eigenvectors: (feature_dim, total_eigen_dim) - each column is an eigenvector
            # param_diff: (batch_size, num_params)
            
            # Make sure we don't use more eigenvectors than available parameters
            num_params_to_use = min(param_diff.shape[1], eigenvectors.shape[1])
            
            # Select the first num_params_to_use eigenvectors (columns)
            selected_eigenvectors = eigenvectors[:, :num_params_to_use]  # (feature_dim, num_params_to_use)
            param_diff_used = param_diff[:, :num_params_to_use]  # (batch_size, num_params_to_use)
            
            # Matrix multiplication: (batch_size, num_params) @ (num_params, feature_dim)^T = (batch_size, feature_dim)
            delta = torch.matmul(param_diff_used, selected_eigenvectors.T)  # (batch_size, feature_dim)
            transformed_features = base_features + delta
            
            # Apply regressor to transformed features
            if opt.method == "Domain":
                processed1_transformed = regressor(transformed_features)
            else:
                processed1_transformed = regressor(transformed_features)
            
            # Calculate distances between transformed f1 and original f2
            distances = torch.norm(processed1_transformed - processed2, p=2, dim=1)
            
            # Store results (move to CPU)
            all_features1.append(features1.cpu() if 'features1' in locals() else processed1.cpu())
            all_features2.append(features2.cpu() if 'features2' in locals() else processed2.cpu())
            all_processed_features1.append(processed1_transformed.cpu())  # Store transformed version
            all_processed_features2.append(processed2.cpu())
            all_cam_params1.append(cam1.cpu())
            all_cam_params2.append(cam2.cpu())
            all_distances.append(distances.cpu())
    
    # Concatenate all results
    features1_all = torch.cat(all_features1, dim=0)
    features2_all = torch.cat(all_features2, dim=0)
    processed1_all = torch.cat(all_processed_features1, dim=0)
    processed2_all = torch.cat(all_processed_features2, dim=0)
    cam_params1_all = torch.cat(all_cam_params1, dim=0)
    cam_params2_all = torch.cat(all_cam_params2, dim=0)
    distances_all = torch.cat(all_distances, dim=0)
    
    # Calculate metrics
    average_distance = torch.mean(distances_all).item()
    std_distance = torch.std(distances_all).item()
    median_distance = torch.median(distances_all).item()
    
    print(f"Retrieval Results (with eigenvector transformation):")
    print(f"Total pairs processed: {len(distances_all)}")
    print(f"Average distance between transformed f1 and f2: {average_distance:.4f}")
    print(f"Standard deviation: {std_distance:.4f}")
    print(f"Median distance: {median_distance:.4f}")
    print(f"Feature shapes - transformed f1: {processed1_all.shape}, f2: {processed2_all.shape}")
    
    # Return comprehensive results
    results = {
        'average_distance': average_distance,
        'std_distance': std_distance,
        'median_distance': median_distance,
        'all_distances': distances_all,
        'features1': processed1_all,
        'features2': processed2_all,
        'cam_params1': cam_params1_all,
        'cam_params2': cam_params2_all
    }
    
    return results