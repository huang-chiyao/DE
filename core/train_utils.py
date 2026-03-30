import argparse
from collections import defaultdict
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
import time

from tools.evaluation.angle import viewpoint_error_2dof, viewpoint_error_4dof

def train(train_loader, model, criterion_list, optimizer, epoch, opt):
    # ---- START EPOCH TIMER ----
    epoch_start_time = time.time()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    all_f = []
    all_labels = []
    model.eval()
    
    # If fix_eigenvector_epoch is set, we will collect features and labels for eigenvector update
    # until the specified epoch. After that, we will fix the eigenvectors and only update
    with torch.no_grad():
        for idx, data_tuple in enumerate(train_loader):
            if idx % 10 == 0: 
                print(f"Collecting features for eigenvector update, batch {idx}")
            image_0, labels = data_tuple
            data_time.update(time.time() - end)
            # Prepare data: concatenate the two augmented images and move to GPU.
            labels = labels.cuda(non_blocking=True)
            image_0 = torch.cat([image_0[0], image_0[1]], dim=0).cuda(non_blocking=True)
            # Forward pass through the model.
            # The updated model returns:
            # f: the features (B, feature)
            f_proj, f, _ = model(image_0)
            all_f.append(f.cpu())
            all_labels.append(labels.cpu())
            
            if opt.max_samples_for_model_update is not None:
                if (idx+1) * opt.batch_size >= opt.max_samples_for_model_update:
                    print(f"Reached max samples for model update: {opt.max_samples_for_model_update}, stopping eigen update data collection.")
                    break
    
    all_f = torch.cat(all_f, dim=0) # (dataset, feat_dim)
    all_labels = torch.cat(all_labels, dim=0) # (dataset, label_dim)
    bsz = all_labels.shape[0]
    
    model.running_mean = model.running_mean.cpu()
    model.prev_evecs = model.prev_evecs.cpu()
    
    print(f"Updating eigenvectors at epoch {epoch}...")
    if opt.fix_eigenvector_epoch is None or epoch < opt.fix_eigenvector_epoch:
        model.get_eigenvectors_and_projection(all_f, train=True, epoch=epoch, update=True)
    else:
        print(f"Fixing eigenvectors at epoch {epoch}. No further updates to eigenvectors will be made.")
        model.get_eigenvectors_and_projection(all_f, train=True, epoch=epoch, update=False)
    
    model.prev_evecs = model.prev_evecs.cuda(non_blocking=True)
    model.running_mean = model.running_mean.cuda(non_blocking=True)

    # ---- END EPOCH TIMER for eigenvector calculation ----
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    print(f"\n eigenvector completed in {epoch_duration / 60:.2f} minutes ({epoch_duration:.2f} seconds).")
    
    model.train()
    
    # Training loop metrics
    total_losses = [0.0] * len(criterion_list)  # Track loss for each criterion
    total_loss = 0.0
    num_batches = 0
    
    for idx, data_tuple in enumerate(train_loader):
        if idx % 10 == 0:
            print(f"start batch {idx}")
        image_0, labels = data_tuple
        data_time.update(time.time() - end)
        # Prepare data: concatenate the two augmented images and move to GPU.
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        image_0 = torch.cat([image_0[0], image_0[1]], dim=0).cuda(non_blocking=True)
        
        # Forward pass through the model.
        # The updated model returns:
        # f_proj: the PCA projected features (B, output_dim)
        # eigenvectors: the aligned principal components (feature_dim, output_dim)
        # regressed_value: the output from the regressor (B, output_dim)
        f_proj, f, proj_coeffs = model(image_0)
        f_proj = torch.cat(torch.split(f_proj.unsqueeze(1), [bsz, bsz], dim=0), dim=1)
        
        # Compute losses using multiple criterion functions
        combined_loss = 0.0
        losses_per_output = []
        
        for i, criterion in enumerate(criterion_list):
            # Get the loss type for this output
            loss_type = opt.encoder_loss_list[i].lower()
            
            if loss_type in ['rnc', 'supcon']:
                dim_loss = criterion(f_proj[:, :, i, :].squeeze(), labels[:, i]) * opt.w_list[i]
            elif loss_type in ['l1', 'l2', 'mse']:
                # For regression losses, use projection coefficients
                dim_loss = criterion(proj_coeffs[:, i].unsqueeze(-1),
                                   torch.cat([labels[:, i], labels[:, i]], dim=0)) * opt.w_list[i]
            elif loss_type in ['crossentropy', 'ce']:
                # For classification losses, use projected features
                dim_loss = criterion(f_proj[:, :, i, :].reshape(2*bsz, -1).squeeze(), torch.cat([labels[:, i], labels[:, i]], dim=0).squeeze().long()) * opt.w_list[i]
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
            combined_loss += dim_loss 
            losses_per_output.append(dim_loss.item())
            total_losses[i] += dim_loss.item()

        loss = combined_loss
        
        losses.update(loss.item(), bsz)
        total_loss += combined_loss.item()
        num_batches += 1
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log batch-level metrics periodically
        if (idx + 1) % opt.print_freq == 0:
            batch_log_dict = {
                "encoder/epoch": epoch,
                "encoder/batch_time": batch_time.val,
                "encoder/data_time": data_time.val,
                "encoder/batch_loss": losses.val,
                "encoder/batch_combined_loss": combined_loss.item(),
                "encoder/batch_idx": idx + 1,
            }
            
            # Log per-output losses
            for i, (output_type, loss_val) in enumerate(zip(opt.output_list, losses_per_output)):
                batch_log_dict[f"encoder/{output_type}_loss"] = loss_val
                batch_log_dict[f"encoder/loss_dim_{i+1}"] = loss_val
            
            wandb.log(batch_log_dict)
            
            if idx < 21:  # Keep the print for first few batches if desired
                print(f'Train: [{epoch}][{idx + 1}/{len(train_loader)}] '
                      f'BT {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'DT {data_time.val:.3f} ({data_time.avg:.3f}) '
                      f'Loss {losses.val:.5f} ({losses.avg:.5f}) '
                      f'Combined {combined_loss:.3f}')
                
                # Print individual losses
                for i, (output_type, loss_val) in enumerate(zip(opt.output_list, losses_per_output)):
                    print(f"  {output_type} loss: {loss_val:.4f}")
        
        if opt.max_samples_for_model_update is not None:
            if (idx+1) * opt.batch_size >= opt.max_samples_for_model_update:
                print(f"Reached max samples for model update: {opt.max_samples_for_model_update}, stopping training for this epoch.")
                break
    
    # Log epoch-level metrics
    epoch_log_dict = {
        "encoder/epoch": epoch,
        "encoder/total_loss": total_loss / num_batches if num_batches > 0 else 0,
    }
    
    # Log average loss for each output type
    for i, output_type in enumerate(opt.output_list):
        avg_loss = total_losses[i] / num_batches if num_batches > 0 else 0
        epoch_log_dict[f"encoder/{output_type}_avg_loss"] = avg_loss
        epoch_log_dict[f"encoder/avg_loss_dim_{i+1}"] = avg_loss
    
    wandb.log(epoch_log_dict)
    # ---- END EPOCH TIMER ----
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    print(f"\nEpoch {epoch} completed in {epoch_duration / 60:.2f} minutes ({epoch_duration:.2f} seconds).")


def train_regressors(train_loader, model, regressor, criterion_list, reg_optimizers, epoch, opt):
    # ---- START EPOCH TIMER ----
    epoch_start_time = time.time()
    model.eval()
    regressor.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Track per-dimension losses for epoch averaging
    epoch_losses_per_dim = [[] for _ in range(opt.output_dim)]
    end = time.time()
    
    # Track total losses for each output type
    total_losses = [0.0] * len(criterion_list)
    num_batches = 0
    
    for idx, data_tuple in enumerate(train_loader):
        image_0, labels = data_tuple
        data_time.update(time.time() - end)
        
        # Prepare data: concatenate the two augmented images and move to GPU.
        # Duplicate labels to match the concatenated images
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        # Concatenate images: [img0, img0_aug] -> shape (2*B, ...)
        image_0 = torch.cat([image_0[0], image_0[1]], dim=0).cuda(non_blocking=True)
        
        # Duplicate labels to match: [label, label] -> shape (2*B, output_dim)
        labels_duplicated = torch.cat([labels, labels], dim=0)
        
        # Forward through frozen extractor
        with torch.no_grad():
            f_proj, f, _ = model(image_0, return_centered=False)  # (2*B, D, feat_dim)
            
            # f_proj, f_centered, _ = model(image_0, return_centered=False)  # (2*B, D, feat_dim)
        
        # Break the graph back to the extractor
        f_proj = f_proj.detach()
        f = f.detach()
        # f_centered = f_centered.detach()
        
        # Collect per-dim losses
        losses_i = []
        predictions_i = []
        batch_total_loss = 0.0
        
        for i in range(opt.output_dim):
            reg_optimizers[i].zero_grad()
            
            # Use full features for dim i → shape (2*B, feat_dim)
            # if opt.output_list[i].lower() in ['classification', 'class', 'cls', 'id']:
            #     print("use centered features for classification decoder at idx ", i)
            #     feat_i = f_centered
            # else:
            feat_i = f
            
            # Decode and compute loss using appropriate criterion
            pred_i = regressor.decoders[i](feat_i)
            
            # Get loss type for this output
            loss_type = opt.decoder_loss_list[i].lower()
            
            if loss_type in ['crossentropy', 'ce']:
                # For classification, convert labels to long
                loss_i = criterion_list[i](pred_i, labels_duplicated[:, i].squeeze().long())
            else:
                # For regression losses (L1, L2, MSE, Huber, etc.)
                loss_i = criterion_list[i](pred_i.squeeze(), labels_duplicated[:, i].squeeze())
            
            # Backward and step
            loss_i.backward()
            reg_optimizers[i].step()
            
            losses_i.append(loss_i.item())
            predictions_i.append(pred_i.detach().cpu())
            epoch_losses_per_dim[i].append(loss_i.item())
            total_losses[i] += loss_i.item()
            batch_total_loss += loss_i.item()
        
        num_batches += 1
        losses.update(batch_total_loss, bsz)
        
        # Print per-dim losses with output type names
        if ((idx + 1) % opt.print_freq == 0 and idx <= 20) or (hasattr(opt, 'debug') and opt.debug):
            loss_str = ' '.join(f'{opt.output_list[i]}={losses_i[i]:.5f}'
                               for i in range(opt.output_dim))
            print(f'Train Regressor: [{epoch}][{idx+1}/{len(train_loader)}] '
                  f'DT {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Total Loss: {batch_total_loss:.5f} '
                  f'{loss_str}')
        
        batch_time.update(time.time() - end)
        end = time.time()
    
    # Log epoch-level metrics
    epoch_log_dict = {"regressor/epoch": epoch}
    
    # Calculate and log epoch averages for each dimension
    total_epoch_loss = 0
    for i in range(opt.output_dim):
        if epoch_losses_per_dim[i]:  # Check if we have losses for this dimension
            avg_loss_dim = np.mean(epoch_losses_per_dim[i])
            epoch_log_dict[f"regressor/{opt.output_list[i]}_loss"] = avg_loss_dim
            epoch_log_dict[f"regressor/loss_dim_{i+1}"] = avg_loss_dim
            total_epoch_loss += avg_loss_dim
    
    epoch_log_dict["regressor/loss_total"] = total_epoch_loss
    epoch_log_dict["regressor/loss_mean"] = total_epoch_loss / opt.output_dim
    
    # Log average loss for each output type
    for i, output_type in enumerate(opt.output_list):
        avg_loss = total_losses[i] / num_batches if num_batches > 0 else 0
        epoch_log_dict[f"regressor/{output_type}_avg_epoch_loss"] = avg_loss
    
    # Log epoch metrics
    wandb.log(epoch_log_dict)

    # ---- END EPOCH TIMER ----
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    print(f"\nEpoch {epoch} completed in {epoch_duration / 60:.2f} minutes ({epoch_duration:.2f} seconds).")

    
def train_joint(train_loader, model, criterion, optimizer, epoch, opt):
    """
    One epoch of joint training for L1_Encoder (which contains its own regressor).
    We take both augmented views (image_pair[0] and image_pair[1]), concatenate them
    along the batch dimension, and feed the 2B images through model. The model's
    forward returns a scalar (or 1-D) prediction for each image. We duplicate each
    label so that both views incur L1 loss against the same scalar target.
    """
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_meter = AverageMeter()
    
    end = time.time()
    for idx, data_tuple in enumerate(train_loader):
        image_pair, labels = data_tuple
        data_time.update(time.time() - end)
        
        # Move labels to GPU; expecting labels shape (B,2)
        labels = labels.cuda(non_blocking=True)
        
        # Concatenate both views along the batch dimension → shape: (2B, C, H, W)
        view1 = image_pair[0].cuda(non_blocking=True)
        view2 = image_pair[1].cuda(non_blocking=True)
        images_both = torch.cat([view1, view2], dim=0)
        
        # Duplicate labels so that each view has the same labels → shape: (2B, 2)
        labels_both = torch.cat([labels, labels], dim=0).squeeze()
        
        # Forward pass through L1_Encoder
        _, _ , pred_both = model(images_both)
        # pred_both has shape (2B, 1)
        
        # Compute L1 loss over both views
        loss = criterion(pred_both, labels_both)
        losses_meter.update(loss.item(), labels_both.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log batch-level metrics during training
        if (idx + 1) % opt.print_freq == 0:
            print(
                f'Train: [{epoch}][{idx+1}/{len(train_loader)}]\t'
                f'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses_meter.val:.5f} ({losses_meter.avg:.5f})'
            )
            
    
    # Log epoch-level metrics
    wandb.log({
        "regressor/epoch": epoch,
        "regressor/loss_total": losses_meter.avg,
    })
    
def train_RNC(train_loader, model, criterion_list, optimizer, epoch, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    
    model.train()
    
    # Training loop metrics
    total_losses = [0.0] * len(criterion_list)  # Track loss for each criterion
    total_loss = 0.0
    num_batches = 0
    
    for idx, data_tuple in enumerate(train_loader):
        image_0, labels = data_tuple
        data_time.update(time.time() - end)
        # Prepare data: concatenate the two augmented images and move to GPU.
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        image_0 = torch.cat([image_0[0], image_0[1]], dim=0).cuda(non_blocking=True)
        
        # Forward pass through the model.
        f_centered, f, _ = model(image_0, return_centered=False)
        f_centered = torch.cat(torch.split(f_centered.unsqueeze(1), [bsz, bsz], dim=0), dim=1)
        
        # Compute losses using multiple criterion functions
        combined_loss = 0.0
        losses_per_output = []
        
        for i, criterion in enumerate(criterion_list):
            # Get the loss type for this output
            loss_type = opt.encoder_loss_list[i].lower()
            
            if loss_type in ['rnc', 'supcon']:
                dim_loss = criterion(f_centered, labels[:, i])
            elif loss_type in ['crossentropy', 'ce']:
                # For classification losses, use projected features
                # dim_loss = criterion(f_proj[:, :, i, :].squeeze(), labels[:, i].long()) 
                raise NotImplementedError("Cross-entropy loss is not implemented in this training loop.")
            
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
            combined_loss += dim_loss
            losses_per_output.append(dim_loss.item())
            total_losses[i] += dim_loss.item()

        loss = combined_loss
        
        losses.update(loss.item(), bsz)
        total_loss += combined_loss.item()
        num_batches += 1
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log batch-level metrics periodically
        if (idx + 1) % opt.print_freq == 0:
            batch_log_dict = {
                "encoder/epoch": epoch,
                "encoder/batch_time": batch_time.val,
                "encoder/data_time": data_time.val,
                "encoder/batch_loss": losses.val,
                "encoder/batch_combined_loss": combined_loss.item(),
                "encoder/batch_idx": idx + 1,
            }
            
            # Log per-output losses
            for i, (output_type, loss_val) in enumerate(zip(opt.output_list, losses_per_output)):
                batch_log_dict[f"encoder/{output_type}_loss"] = loss_val
                batch_log_dict[f"encoder/loss_dim_{i+1}"] = loss_val
            
            wandb.log(batch_log_dict)
            
            if idx < 21:  # Keep the print for first few batches if desired
                print(f'Train: [{epoch}][{idx + 1}/{len(train_loader)}] '
                      f'BT {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'DT {data_time.val:.3f} ({data_time.avg:.3f}) '
                      f'Loss {losses.val:.5f} ({losses.avg:.5f}) '
                      f'Combined {combined_loss:.3f}')
                
                # Print individual losses
                for i, (output_type, loss_val) in enumerate(zip(opt.output_list, losses_per_output)):
                    print(f"  {output_type} loss: {loss_val:.4f}")
        
    # Log epoch-level metrics
    epoch_log_dict = {
        "encoder/epoch": epoch,
        "encoder/total_loss": total_loss / num_batches if num_batches > 0 else 0,
    }
    
    # Log average loss for each output type
    for i, output_type in enumerate(opt.output_list):
        avg_loss = total_losses[i] / num_batches if num_batches > 0 else 0
        epoch_log_dict[f"encoder/{output_type}_avg_loss"] = avg_loss
        epoch_log_dict[f"encoder/avg_loss_dim_{i+1}"] = avg_loss
    
    wandb.log(epoch_log_dict)

def test(test_loader, model, regressor, opt, device="cpu", val=False, epoch=None):
    model.eval()
    regressor.eval()
    
    # (1) Make sure model + regressor are on the correct device
    device = torch.device(device)
    model.to(device)
    regressor.to(device)
    
    all_outputs = defaultdict(list)
    all_labels = []
    all_features = []
    all_coefficients = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image_batch, labels_batch = batch  # always 2 elements returned
            inputs = image_batch[0].to(device, non_blocking=True)

            labels_batch = labels_batch  # shape: (batch_size, D)
            labels = labels_batch.to(device, non_blocking=True)
            
            # Forward pass through the encoder + regressor
            if opt.method == "Domain":
                f_proj, f, coeffs = model(inputs, return_centered=False)
                # f_proj, f_centered, _ = model(inputs, return_centered=False)
                
                # All outputs (regression + classification) processed together
                for i in range(opt.output_dim):
                    # if opt.output_list[i].lower() in ['classification', 'class', 'cls', 'id']:
                    #     print("use centered features for classification decoder at idx ", i)
                    #     all_outputs[i].append(regressor.decoders[i](f).cpu())
                    # else:
                    all_outputs[i].append(regressor.decoders[i](f).cpu())
                
                all_labels.append(labels.cpu())
                all_features.append(f_proj.cpu())
                all_coefficients.append(coeffs.cpu())
                
            elif opt.method == "RNC":
                f_center, f, _ = model(inputs)
                
                # All outputs processed together
                for i in range(opt.output_dim):
                    all_outputs[i].append(regressor.decoders[i](f).cpu())
                all_labels.append(labels.cpu())
                
            elif opt.method == "L1":
                # TODO: implement for L1 is not done yet
                f = model(inputs)
                
                # All outputs processed together
                processed = regressor(f)
                all_outputs.append(processed.cpu())
                all_labels.append(labels.cpu())

            else:
                raise ValueError(f"Your method is not supported: {opt.method}")
    
    # (2) Concatenate across batches
    for i in range(opt.output_dim):
        all_outputs[i] = torch.cat(all_outputs[i], dim=0)
        print(f"Output {i} shape: {all_outputs[i].shape}")
    labels = torch.cat(all_labels, dim=0)    # same shape
    print(f"Shapes — labels: {labels.shape}")
    
    # Determine logging prefix and step metric based on validation or test
    prefix = "val" if val else "test"
    get_metric(prefix, all_outputs, labels, opt, epoch=epoch)

    
def get_metric(prefix, outputs, labels, opt, epoch=None):
    step_metric = f"{prefix}/epoch"
    # Use epoch if provided, otherwise use step 1
    step_value = epoch if epoch is not None else 1
    
    # Skip if no outputs
    if opt.output_dim == 0 or outputs is None or labels is None:
        print("Skipping metrics (no outputs available)")
        return
    
    # Separate outputs by type (regression vs classification)
    regression_indices = []
    classification_indices = []
    regression_outputs = []
    regression_labels = []
    classification_outputs = []
    classification_labels = []
    
    for i, output_type in enumerate(opt.output_list):
        is_classification = any(keyword in output_type.lower() for keyword in opt.classification_keywords)
        
        if is_classification:
            classification_indices.append(i)
            classification_outputs.append(outputs[i])
            classification_labels.append(labels[:, i].squeeze())
        else:
            regression_indices.append(i)
            regression_outputs.append(outputs[i])  # Keep as 2D tensor
            regression_labels.append(labels[:, i].unsqueeze(-1))  # Keep as 2D tensor
            
    # Handle classification metrics
    if classification_indices:
        print(f"Processing {len(classification_indices)} classification outputs")
        
        for i, (cls_out, cls_lab) in enumerate(zip(classification_outputs, classification_labels)):
            # Find the corresponding output type name
            cls_output_name = f"classification_{i}"
            
            # Get the corresponding loss type
            loss_type = opt.decoder_loss_list[classification_indices[i]].lower()
            print(f"Processing {cls_output_name} with loss type '{loss_type}', decoder_loss_list: {opt.decoder_loss_list}")
            
            if loss_type in ['crossentropy', 'ce']:
                # For cross-entropy: outputs are logits, use argmax for predictions
                cls_predictions = torch.argmax(cls_out, dim=-1)
                cls_loss = F.cross_entropy(cls_out, cls_lab.long()).item()
                num_classes = cls_out.shape[-1]
            elif loss_type == 'l1':
                # For L1: output is a single value, round to nearest integer for predictions
                cls_predictions = torch.round(cls_out.squeeze()).long()
                # Clamp predictions to valid class range [0, num_cls-1]
                num_classes = int(cls_lab.max().item()) + 1  # Infer from labels
                cls_predictions = torch.clamp(cls_predictions, 0, num_classes - 1)
                cls_loss = F.l1_loss(cls_out.squeeze(), cls_lab.float()).item()
            else:
                # Default to cross-entropy
                print(f"Warning: Unknown loss type '{loss_type}' for classification, using cross-entropy")
                cls_predictions = torch.argmax(cls_out, dim=-1)
                cls_loss = F.cross_entropy(cls_out, cls_lab.long()).item()
                num_classes = cls_out.shape[-1]
            
            # Calculate classification accuracy
            correct_predictions = (cls_predictions.squeeze() == cls_lab.squeeze()).float()
            cls_accuracy = correct_predictions.mean().item()
            
            # Log classification metrics
            wandb.log({
                step_metric: step_value,
                f"{prefix}/{cls_output_name}_accuracy": cls_accuracy,
                f"{prefix}/{cls_output_name}_loss": cls_loss,
                f"{prefix}/{cls_output_name}_num_classes": num_classes,
            })
            
            print(f"{cls_output_name} Accuracy: {cls_accuracy:.4f}, Loss: {cls_loss:.4f}")
    
    # Handle regression metrics
    if not regression_indices:
        print("No regression outputs to evaluate")
        return
    
    print(f"Processing {len(regression_indices)} regression outputs")
    
    # Combine all regression outputs
    if regression_outputs:
        reg_outputs = torch.cat(regression_outputs, dim=1)
        reg_labels = torch.cat(regression_labels, dim=1).squeeze(-1)  # Ensure labels are 2D
    else:
        print("No regression data available")
        return
    
    if opt.dataset in ['ShapeNetRenderingDataset']:
        # Determine output type based on regression outputs and output_list
        regression_output_types = []
        for output_type in opt.output_list:
            is_classification = any(keyword in output_type.lower() for keyword in opt.classification_keywords)
            if not is_classification:
                regression_output_types.append(output_type)
        
        # Function to reorder and group outputs
        def reorder_and_group_outputs(output_types, outputs, labels):
            """Reorder outputs to standard format and group by evaluation type"""
            output_types_lower = [ot.lower() for ot in output_types]
            
            # Define standard orderings
            angle_order = ['az', 'el', 'ro']  # or 'ip' instead of 'ro'
            trig_order = ['sinaz', 'cosaz', 'sinel', 'cosel', 'sinro', 'cosro']
            
            grouped_evaluations = []
            used_indices = set()
            
            # Check for trigonometric representation
            trig_indices = {}
            for trig_name in trig_order:
                for i, ot in enumerate(output_types_lower):
                    if trig_name in ot and i not in used_indices:
                        trig_indices[trig_name] = i
                        break
            
            # If we have trigonometric outputs, group them
            if len(trig_indices) >= 4:  # At least sinaz, cosaz, sinel, cosel
                trig_reordered_indices = []
                trig_names_found = []
                
                for trig_name in trig_order:
                    if trig_name in trig_indices:
                        trig_reordered_indices.append(trig_indices[trig_name])
                        trig_names_found.append(trig_name)
                        used_indices.add(trig_indices[trig_name])
                
                if trig_reordered_indices:
                    trig_outputs = outputs[:, trig_reordered_indices]
                    trig_labels = labels[:, trig_reordered_indices]
                    grouped_evaluations.append(('trig', trig_outputs, trig_labels, len(trig_reordered_indices)))
            
            # Check for direct angle representation
            angle_indices = {}
            for angle_name in angle_order:
                for i, ot in enumerate(output_types_lower):
                    if i not in used_indices:
                        # Check for exact match or common variations
                        if (angle_name == ot or 
                            (angle_name == 'ro' and ('ip' in ot or 'roll' in ot)) or
                            (angle_name == 'az' and ('azimuth' in ot or 'yaw' in ot)) or
                            (angle_name == 'el' and ('elevation' in ot or 'pitch' in ot))):
                            angle_indices[angle_name] = i
                            break
            
            # If we have direct angle outputs, group them
            if len(angle_indices) >= 2:  # At least az, el
                angle_reordered_indices = []
                angle_names_found = []
                
                for angle_name in angle_order:
                    if angle_name in angle_indices:
                        angle_reordered_indices.append(angle_indices[angle_name])
                        angle_names_found.append(angle_name)
                        used_indices.add(angle_indices[angle_name])
                
                if angle_reordered_indices:
                    angle_outputs = outputs[:, angle_reordered_indices]
                    angle_labels = labels[:, angle_reordered_indices]
                    grouped_evaluations.append(('angle', angle_outputs, angle_labels, len(angle_reordered_indices)))
            
            # Handle any remaining unused outputs as default
            remaining_indices = [i for i in range(outputs.shape[1]) if i not in used_indices]
            if remaining_indices:
                remaining_outputs = outputs[:, remaining_indices]
                remaining_labels = labels[:, remaining_indices]
                grouped_evaluations.append(('default', remaining_outputs, remaining_labels, len(remaining_indices)))
            
            return grouped_evaluations
        
        # Group and reorder outputs
        grouped_evaluations = reorder_and_group_outputs(regression_output_types, reg_outputs, reg_labels)
        
        # Evaluate each group separately
        all_metrics = {}
        
        for eval_idx, (eval_type, outputs, labels, num_dims) in enumerate(grouped_evaluations):
            group_prefix = f"{prefix}_group{eval_idx}_{eval_type}" if len(grouped_evaluations) > 1 else prefix
            
            if eval_type == 'trig':
                # Trigonometric representation evaluation
                if num_dims == 6:
                    # sin_az, cos_az, sin_el, cos_el, sin_ro, cos_ro (6DOF)
                    mae_error = calculate_mae(outputs, labels)
                    errors = viewpoint_error_6dof(outputs, labels)
                    num_dimension_display = 6
                    eval_description = "6DOF trigonometric (sinaz, cosaz, sinel, cosel, sinro, cosro)"
                elif num_dims == 4:
                    # sin_az, cos_az, sin_el, cos_el (4DOF)
                    mae_error = calculate_mae(outputs, labels)
                    errors = viewpoint_error_4dof(outputs, labels)
                    num_dimension_display = 4
                    eval_description = "4DOF trigonometric (sinaz, cosaz, sinel, cosel)"
                else:
                    mae_error = calculate_mae(outputs, labels)
                    errors = torch.norm(outputs - labels, dim=1)
                    num_dimension_display = num_dims
                    eval_description = f"{num_dims}DOF trigonometric"
                    
            elif eval_type == 'angle':
                # Direct angle representation evaluation
                if num_dims == 3 and any('az' in regression_output_types[0].lower() and 'el' in regression_output_types[1].lower() for _ in [None]):
                    # az, el, ro/ip (3DOF)
                    mae_error = calculate_circular_mae(outputs, labels)
                    errors = viewpoint_error_3dof(outputs, labels)
                    num_dimension_display = 3
                    eval_description = "3DOF angles (az, el, ro/ip)"
                elif num_dims == 2:
                    # az, el (2DOF)
                    mae_error = calculate_circular_mae(outputs, labels)
                    errors = viewpoint_error_2dof(outputs, labels)
                    num_dimension_display = 2
                    eval_description = "2DOF angles (az, el)"
                else:
                    mae_error = calculate_circular_mae(outputs, labels)
                    errors = torch.norm(outputs - labels, dim=1)
                    num_dimension_display = num_dims
                    eval_description = f"{num_dims}DOF angles"
                    
            else:
                # Default evaluation for remaining outputs
                mae_error = calculate_mae(outputs, labels)
                errors = torch.norm(outputs - labels, dim=1)
                num_dimension_display = num_dims
                eval_description = f"{num_dims}DOF default"
            
            # Calculate metrics
            median_err = errors.median().item()
            threshold_30 = 30.0 * torch.pi / 180.0  # 30 degrees to radians
            threshold_15 = 15.0 * torch.pi / 180.0  # 15 degrees to radians
            acc30 = (errors < threshold_30).float().mean().item()
            acc15 = (errors < threshold_15).float().mean().item()
            
            # Store metrics
            group_metrics = {
                f"{group_prefix}/median_error_deg": median_err * 180 / torch.pi,
                f"{group_prefix}/accuracy_30deg": acc30,
                f"{group_prefix}/accuracy_15deg": acc15,
            }
            
            # Add MAE for each dimension
            for i in range(min(num_dimension_display, len(mae_error))):
                group_metrics[f"{group_prefix}/mae_dim{i}"] = mae_error[i].item()
            
            all_metrics.update(group_metrics)
            
            # Print results for this group
            mae_str = ", ".join([f"MAE_dim{i}: {mae_error[i]:.4f}" for i in range(min(len(mae_error), num_dimension_display))])
            print(f"[{eval_description}] Median Error: {median_err * 180 / torch.pi:.2f}°, Acc@30°: {acc30:.4f}, {mae_str}")
        
        # Log all metrics with proper step metric
        all_metrics[step_metric] = step_value
        wandb.log(all_metrics)
    
    else:
        # Generic regression evaluation
        mae_error = calculate_mae(reg_outputs, reg_labels)
        mse_error = F.mse_loss(reg_outputs, reg_labels, reduction='none').mean(dim=0)
        
        wandb.log({
            step_metric: step_value,
            **{f"{prefix}/mae_dim{i}": mae_error[i].item() for i in range(len(mae_error))},
            **{f"{prefix}/mse_dim{i}": mse_error[i].item() for i in range(len(mse_error))},
            f"{prefix}/regression_output_dim": len(regression_indices)
        })
        
        mae_str = ", ".join([f"MAE_dim{i}: {mae_error[i]:.4f}" for i in range(len(mae_error))])
        print(f"Regression metrics: {mae_str}")

def test_retrivial(test_loader, model, regressor, opt, device="cuda", wandb_log=True):
    """
    Updated retrieval test function using decoder inverse mapping
    Handles mixed loss types per decoder based on decoder_loss_list
    """
    model.eval()
    regressor.eval()
    
    # Make sure model + regressor are on the correct device
    model.to(device)
    regressor.to(device)
    
    # Parse decoder loss types from opt.decoder_loss_list
    decoder_losses = opt.decoder_loss_list
    print(f"Decoder loss types: {decoder_losses}")
    
    # Collect all data first
    all_features = []
    all_labels = []
    all_coefficients = []
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image_batch, labels_batch = batch  # New dataloader format
            inputs = image_batch[0].to(device, non_blocking=True)
            labels = labels_batch.to(device, non_blocking=True)
            
            # Forward pass through the encoder
            if opt.method == "Domain":
                f_proj, f, coeffs = model(inputs)
                all_features.append(f.cpu())  # Use intermediate features f
                all_coefficients.append(coeffs.cpu())
            elif opt.method == "RNC":
                f_center, f, _ = model(inputs)
                all_features.append(f.cpu())
            elif opt.method == "L1":
                f = model(inputs)
                all_features.append(f.cpu())
            else:
                raise ValueError(f"Your method is not supported: {opt.method}")
            
            all_labels.append(labels.cpu())
    
    # Concatenate all data
    features_all = torch.cat(all_features, dim=0)  # (total_samples, feature_dim)
    labels_all = torch.cat(all_labels, dim=0)      # (total_samples, num_outputs, output_dim) or (total_samples, total_dims)
    
    print(f"Shapes — features: {features_all.shape}, labels: {labels_all.shape}")
    labels_all = labels_all.squeeze() # (total_samples, num_outputs)
    
    # Split into two halves: prediction and ground_truth
    total_samples = features_all.shape[0]
    half_point = int(total_samples // 2)
    
    # First half for "prediction", second half for "ground_truth"
    f_prediction = features_all[:half_point].to(device)        # (half_samples, feature_dim)
    labels_prediction = labels_all[:half_point].to(device)     # (half_samples, total_output_dims)
    
    f_ground_truth = features_all[half_point:half_point*2].to(device)  # (half_samples, feature_dim)
    labels_ground_truth = labels_all[half_point:half_point*2].to(device) # (half_samples, total_output_dims)
    
    print(f"Split data - Prediction: {f_prediction.shape}, Ground truth: {f_ground_truth.shape}")
    
    # Get decoder results from prediction features - handle mixed loss types
    decoder_outputs = []
    
    if opt.method == "L1":
        # # For L1 method, get the full combined output once
        # full_decoder_output = regressor(f_prediction)  # (half_samples, total_combined_dims)
        
        # # Split by output dimensions based on loss types
        # current_idx = 0
        # for i, loss_type in enumerate(decoder_losses):
        #     if loss_type.lower() == 'ce':
        #         # CE decoder outputs num_classes dimensions
        #         num_classes = getattr(opt, 'num_classes', 10)
        #         decoder_result = full_decoder_output[:, current_idx:current_idx + num_classes]
        #         current_idx += num_classes
        #     else:
        #         # L1/L2 decoder outputs 1 dimension
        #         decoder_result = full_decoder_output[:, current_idx:current_idx + 1]
        #         current_idx += 1
            
        #     decoder_outputs.append(decoder_result)
        raise NotImplementedError("Mixed loss types for L1 method is not implemented yet.")
            
    else:
        # For Domain/RNC methods, each decoder is separate
        for i, loss_type in enumerate(decoder_losses):
            decoder_result = regressor.decoders[i](f_prediction)  
            decoder_outputs.append(decoder_result)
    
    print(f"Decoder outputs shapes: {[out.shape for out in decoder_outputs]}")
    
    # Process each decoder output based on its loss type
    modified_labels_list = []
    
    for i, (loss_type, decoder_output) in enumerate(zip(decoder_losses, decoder_outputs)):
        print(f"Processing decoder {i}: loss_type={loss_type}, decoder_output.shape={decoder_output.shape}")
        
        print(f"decoder_output (before): {decoder_output[3, :]}")
        if loss_type.lower() == 'ce':
            labels_gt_int = labels_ground_truth[:, i].long()
            labels_pred_int = labels_prediction[:, i].long()
            num_classes = decoder_output.shape[-1]

            # Decoder probabilities (softmax)
            exp_logits = torch.exp(decoder_output)
            b = exp_logits.sum(dim=-1, keepdim=True)
            decoder_probs = exp_logits / b

            # Copy for modification
            modified_probs = decoder_probs.clone()
            print(f"modified_probs (before): {modified_probs[3, :]}")
            print(f"labels_gt_int[3]: {labels_gt_int[3]}, labels_pred_int[3]: {labels_pred_int[3]}")

            # Gather the values before swap
            gt_vals = decoder_probs[torch.arange(decoder_probs.size(0)), labels_gt_int]
            pred_vals = decoder_probs[torch.arange(decoder_probs.size(0)), labels_pred_int]

            # Swap ground truth and predicted probabilities
            modified_probs[torch.arange(modified_probs.size(0)), labels_gt_int] = pred_vals # 1
            modified_probs[torch.arange(modified_probs.size(0)), labels_pred_int] = gt_vals # 0
            # modified_probs[torch.arange(modified_probs.size(0)), labels_gt_int] = 1 # 1
            # modified_probs[torch.arange(modified_probs.size(0)), labels_pred_int] = 0 # 0
            # modified_probs = torch.zeros_like(decoder_probs)
            # modified_probs.scatter_(1, labels_gt_int.unsqueeze(1), 1.0)

            print(f"modified_probs (after swap): {modified_probs[3, :]}")
            # Re-normalize to make sure it is still a distribution
            modified_probs = modified_probs / (modified_probs.sum(dim=-1, keepdim=True) + 1e-8)

            # Multiply back by denominator b if you want to keep the "logits-like" structure
            modified_labels = modified_probs * b
            modified_labels = torch.log(modified_labels + 1e-8)  # Add small value to avoid log(0)
        else:
            delta = labels_ground_truth[:, i].reshape(-1, 1) - labels_prediction[:, i].reshape(-1, 1)
            modified_labels = decoder_output + delta
            print(f"labels_ground_truth[:, {i}] (sample 3): {labels_ground_truth[3, i]}")
            print(f"labels_prediction[:, {i}] (sample 3): {labels_prediction[3, i]}")
         
        print(f"modified_labels (after): {modified_labels[3, :]}")  
        print(f"modified_labels shape: {modified_labels.shape}") 
        modified_labels_list.append(modified_labels)
        print("-"*30)
        
    modified_labels_list = torch.cat(modified_labels_list, dim=-1)  # (half_samples, total_output_dims)
    
    print(f"Modified labels shape after processing all decoders: {modified_labels_list.shape}")
    
    # Concatenate all processed outputs
    all_results = {}
    
    for inverse_method in ["pseudoinverse", "least_squares", "ridge"]:
        print(f"\n{'='*50}")
        print(f"Testing inverse method: {inverse_method.upper()}")
        print(f"{'='*50}")
        
        if opt.method == "Domain" or opt.method == "RNC":
            # Concatenate all decoder weights and biases
            all_weights = []
            all_biases = []
            
            for i, loss_type in enumerate(decoder_losses):
                decoder = regressor.decoders[i]
                all_weights.append(decoder.weight)  # Shape varies: (1, feature_dim) or (num_classes, feature_dim)
                
                bias = decoder.bias if decoder.bias is not None else torch.zeros(decoder.weight.shape[0], device=device)
                all_biases.append(bias)
            
            # Concatenate weights and biases
            combined_weights = torch.cat(all_weights, dim=0)  # (total_output_dims, feature_dim)
            combined_biases = torch.cat(all_biases, dim=0)    # (total_output_dims,)
            
            
        elif opt.method == "L1":
            # For L1, the combined weight matrix is not straightforward
            raise NotImplementedError("Inverse methods for L1 method is not implemented yet.")
        
        print(f"Combined weights shape: {combined_weights.shape}, Combined biases shape: {combined_biases.shape}")
    
        # Center the labels by removing bias
        y_centered = modified_labels_list - combined_biases.unsqueeze(0)  # (half_samples, total_output_dims)
        
        # Apply the current inverse method
        f_reconstructed = None
        method_success = True
        
        try:
            if inverse_method == "pseudoinverse":
                # Method 1: Pseudoinverse of combined weight matrix
                weight_pinv = torch.pinverse(combined_weights)  # (feature_dim, total_output_dims)
                f_reconstructed = torch.matmul(y_centered, weight_pinv.T)  # (half_samples, feature_dim)
                
            elif inverse_method == "least_squares":
                # Method 2: Least squares solution for combined system
                WWT = combined_weights @ combined_weights.T  # (total_output_dims, total_output_dims)
                WWT_inv = torch.inverse(WWT)
                f_reconstructed = y_centered @ WWT_inv @ combined_weights  # (half_samples, feature_dim)
                
            elif inverse_method == "ridge":
                # Method 3: Ridge regression for combined system
                lambda_reg = 1e-6
                WWT_reg = combined_weights @ combined_weights.T + lambda_reg * torch.eye(combined_weights.shape[0], device=device)
                WWT_reg_inv = torch.inverse(WWT_reg)
                f_reconstructed = y_centered @ WWT_reg_inv @ combined_weights  # (half_samples, feature_dim)
                
        except Exception as e:
            print(f"Error in {inverse_method}: {str(e)}")
            method_success = False
        
        if not method_success or f_reconstructed is None:
            print(f"Skipping {inverse_method} due to failure")
            continue
            
        print(f"  Processed {len(decoder_losses)} decoders simultaneously using {inverse_method} method")
        print(f"  Loss types: {decoder_losses}")
        
        # Fix f_reconstructed by eigenvector projection
        f_reconstructed = f_reconstructed - model.running_mean
        coeffs = torch.matmul(f_reconstructed, model.prev_evecs)  # (batch_size, num_dims)
        f_reconstructed = coeffs.unsqueeze(-1) * model.prev_evecs.t().unsqueeze(0)  # (batch_size, num_dims, latent_dim)
        f_reconstructed = f_reconstructed.sum(dim=1)
        f_reconstructed = f_reconstructed + model.running_mean
        
        # Calculate cosine similarity between reconstructed features and ground truth features
        f_reconstructed_norm = torch.nn.functional.normalize(f_reconstructed, p=2, dim=1)
        f_ground_truth_norm = torch.nn.functional.normalize(f_ground_truth, p=2, dim=1)
        
        cosine_similarities = torch.sum(f_reconstructed_norm * f_ground_truth_norm, dim=1)  # (half_samples,)
        l2_distances = torch.norm(f_reconstructed - f_ground_truth, p=2, dim=1)  # (half_samples,)
        
        # Move results to CPU for analysis
        cosine_similarities = cosine_similarities.cpu()
        l2_distances = l2_distances.cpu()
        
        # Calculate metrics
        avg_cosine_sim = torch.mean(cosine_similarities).item()
        std_cosine_sim = torch.std(cosine_similarities).item()
        
        avg_l2_distance = torch.mean(l2_distances).item()
        std_l2_distance = torch.std(l2_distances).item()
        
        # Store results for this method
        method_results = {
            'avg_cosine_similarity': avg_cosine_sim,
            'std_cosine_similarity': std_cosine_sim,
            'avg_l2_distance': avg_l2_distance,
            'std_l2_distance': std_l2_distance,
            'all_cosine_similarities': cosine_similarities,
            'all_l2_distances': l2_distances,
        }
        all_results[inverse_method] = method_results
        
        # Print results for this method
        print(f"\nRetrieval Results for {inverse_method.upper()}:")
        print(f"Total pairs processed: {len(cosine_similarities)}")
        print(f"\nCosine Similarity Metrics:")
        print(f"  Average: {avg_cosine_sim:.4f}")
        print(f"  Std Dev: {std_cosine_sim:.4f}")
        print(f"\nL2 Distance Metrics:")
        print(f"  Average: {avg_l2_distance:.4f}")
        print(f"  Std Dev: {std_l2_distance:.4f}")
        
        # Log to wandb if enabled
        epoch = 1
        
        # Create a suffix based on decoder loss types for logging
        loss_suffix = "_mixed" if len(set(decoder_losses)) > 1 else f"_{decoder_losses[0]}"
        
        # wandb.log({
        #     f"test_retrieval{loss_suffix}/{inverse_method}_avg_cosine_similarity": avg_cosine_sim,
        #     f"test_retrieval{loss_suffix}/{inverse_method}_std_cosine_similarity": std_cosine_sim,
        #     f"test_retrieval{loss_suffix}/{inverse_method}_avg_l2_distance": avg_l2_distance,
        #     f"test_retrieval{loss_suffix}/{inverse_method}_std_l2_distance": std_l2_distance,
        #     f"test_retrieval{loss_suffix}/epoch": epoch
        # })
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON OF ALL INVERSE METHODS")
    print(f"Decoder losses: {decoder_losses}")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Avg Cosine':<12} {'Avg L2':<12} {'Median Cosine':<15} {'Median L2':<12}")
    print("-" * 80)
    
    for method, results in all_results.items():
        print(f"{method:<15} {results['avg_cosine_similarity']:<12.4f} {results['avg_l2_distance']:<12.4f} ")
    
    # Log summary metrics to wandb
    epoch = 1
    
    best_cosine_method = max(all_results.keys(), key=lambda x: all_results[x]['avg_cosine_similarity'])
    best_l2_method = min(all_results.keys(), key=lambda x: all_results[x]['avg_l2_distance'])
    
    loss_suffix = "_mixed" if len(set(decoder_losses)) > 1 else f"_{decoder_losses[0]}"
    
    # wandb.log({
    #     f"test_retrieval{loss_suffix}/best_cosine_score": all_results[best_cosine_method]['avg_cosine_similarity'],
    #     f"test_retrieval{loss_suffix}/best_l2_score": all_results[best_l2_method]['avg_l2_distance'],
    #     f"test_retrieval{loss_suffix}/num_methods_tested": len(all_results),
    #     f"test_retrieval{loss_suffix}/epoch": epoch
    # })
    
    return all_results