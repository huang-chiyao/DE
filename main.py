import argparse
import copy
import os
import sys
import logging
import torch
import torch.nn.functional as F
import time
from core.parser_utils import parse_option, resume_from_wandb_id, init_wandb
from core.train_utils import test_retrivial, train, train_RNC, train_regressors, test
from core.utils import adjust_learning_rate, log_metrics_to_wandb, plot_latent, spearman_test, set_loader, plot_3d_latent, measure_train_memory
from core.model import Domain_Encoder, RNC_Encoder, Regressors
from core.loss import CELoss, RnCLoss, SupConLoss
import wandb

def main():
    opt = parse_option()
    
    # Handle wandb resume by ID
    ckpt_info = None
    if opt.wandb_id:  # Simplified check
        opt, ckpt_info = resume_from_wandb_id(opt, opt.wandb_id)
    else:
        opt = init_wandb(opt)  # Initialize wandb if no ID is provided
    
    # Define wandb metrics to avoid step conflicts
    wandb.define_metric("encoder/epoch")
    wandb.define_metric("regressor/epoch") 
    wandb.define_metric("val/epoch")
    wandb.define_metric("test/epoch")
    wandb.define_metric("test_retrieval/epoch")
    wandb.define_metric("test_noise/epoch")
    wandb.define_metric("encoder_eigenvector/epoch")
    wandb.define_metric("encoder_visualization/epoch")

    # Define metric groups with their respective step metrics
    wandb.define_metric("encoder/*", step_sync=False, step_metric="encoder/epoch")
    wandb.define_metric("regressor/*", step_sync=False, step_metric="regressor/epoch")
    wandb.define_metric("val/*", step_sync=False, step_metric="val/epoch")
    wandb.define_metric("test/*", step_sync=False, step_metric="test/epoch")
    wandb.define_metric("test_retrieval/*", step_sync=False, step_metric="test_retrieval/epoch")
    wandb.define_metric("test_noise/*", step_sync=False, step_metric="test_noise/epoch")
    wandb.define_metric("encoder_eigenvector/*", step_sync=False, step_metric="encoder_eigenvector/epoch")
    wandb.define_metric("encoder_visualization/*", step_sync=False, step_metric="encoder_visualization/epoch")

    # 1) Build data loaders
    train_loader, valid_loader, test_loader = set_loader(opt)

    # 2) Build (or load) the encoder
    in_channels = 1 if "MNIST" in opt.dataset else in_channels
    if opt.method == "Domain":
        model = Domain_Encoder(opt, in_channels=in_channels).cuda() 
    elif opt.method == "RNC":
        model = RNC_Encoder(opt, in_channels=in_channels).cuda()
    else:
        raise ValueError(f"Unsupported method: {opt.method}")

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == 'adam':  # Default to Adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt.optimizer}")
    
    # Create loss function list based on encoder_loss_list
    criterion_list = []

    # Validate input consistency
    if len(opt.encoder_loss_list) != len(opt.output_list) or len(opt.output_list) != len(opt.temp_list) or len(opt.output_list) != len(opt.w_list):
        raise ValueError("encoder_loss_list and output_list and temp_list and w_list must have the same length")

    print(f"Creating loss functions for outputs: {opt.output_list}")
    print(f"Using loss types: {opt.encoder_loss_list}")

    for i, (output_type, loss_type, temp) in enumerate(zip(opt.output_list, opt.encoder_loss_list, opt.temp_list)):
        print(f"Output {i}: {output_type} -> Loss: {loss_type}")
        
        if loss_type.lower() == "l1":
            criterion = torch.nn.L1Loss().cuda()
            print(f"  Created L1Loss for {output_type} and weight {opt.w_list[i]}")
            
        elif loss_type.lower() == "l2" or loss_type.lower() == "mse":
            criterion = torch.nn.MSELoss().cuda()
            print(f"  Created MSELoss for {output_type} and weight {opt.w_list[i]}")
            
        elif loss_type.lower() == "rnc":
            criterion = RnCLoss(
                temperature=temp,
                label_diff=opt.label_diff,
                feature_sim=opt.feature_sim
            ).cuda()
            print(f"  Created RnCLoss for {output_type} and temp {temp} and weight {opt.w_list[i]}")
                
        elif loss_type.lower() == "supcon":
            criterion = SupConLoss(temperature=temp).cuda()
            print(f"  Created SupConLoss for {output_type} and temp {temp} and weight {opt.w_list[i]}")
            
        elif loss_type.lower() == "crossentropy" or loss_type.lower() == "ce":
            criterion = CELoss(model.feature_dim, num_classes=train_loader.dataset.num_classes).cuda()
            print(f"  Created CrossEntropyLoss for {output_type} and weight {opt.w_list[i]}")
            
        else:
            raise ValueError(f"Unknown loss type: {loss_type} for output: {output_type}")
        
        criterion_list.append(criterion)

    print(f"Created {len(criterion_list)} loss functions total")
            
    start_epoch = 1
    encoder_completed = False

    # Initialize best model tracking
    best_avg_spearman = -float('inf')
    best_model = None
    best_epoch = 0

    # Load encoder if available
    if opt.ckpt and os.path.isfile(opt.ckpt):
        ckpt_data = torch.load(opt.ckpt, map_location='cpu')
        model.load_state_dict(ckpt_data['model_state_dict'])
        if 'optimizer_state_dict' in ckpt_data:
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        start_epoch = ckpt_data.get('epoch', 0) + 1
        
        print(f"Loaded encoder from {opt.ckpt}, resuming at epoch {start_epoch}")
        
        # Check if encoder training is completed
        if ckpt_info and ckpt_info['has_regressor']:
            # If we have regressor checkpoints, encoder is likely completed
            encoder_completed = True
            print("Encoder training appears to be completed, will focus on regressor training")
        elif start_epoch > opt.epochs:
            encoder_completed = True
            print("Encoder training epochs exceeded, marking as completed")
    else:
        print(f'Training encoder from scratch')

    # Load best model information with proper error handling
    best_path = os.path.join(opt.save_folder, 'best_encoder.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model = copy.deepcopy(model)
        best_epoch = checkpoint.get('best_epoch', 0)
        print(f"Best model tracking initialized from {best_path} at epoch {best_epoch}")
    else:
        print("Starting fresh best model tracking")

    # Train encoder only if not completed
    if not encoder_completed:
        print(f"Starting encoder training from epoch {start_epoch} to {opt.epochs}")
        # TODO: update latent space visualization in case multiple dimensions
        if sum(opt.output_dimension_list) == opt.output_dim:
            plot_latent(valid_loader, model, 0, opt)
            # plot_latent(valid_loader, model, 0, opt, plot_3d=True). # disable 3d plot for faster training start
        
        for epoch in range(start_epoch, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)
            
            # Log learning rate with proper epoch metric
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "encoder/epoch": epoch,
                "encoder/learning_rate": current_lr,
            })

            if opt.method == "Domain":
                measure_train_memory(train, train_loader, model, criterion_list, optimizer, epoch, opt)
            elif opt.method == "RNC":
                measure_train_memory(train_RNC, train_loader, model, criterion_list, optimizer, epoch, opt)
            
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(save_dict, save_file)
                print(f"Saved encoder checkpoint: {save_file}")

            # Optional: run validation on the encoder's PCA‐features
            if opt.eval_freq > 0 and (epoch % opt.eval_freq == 0 or epoch == opt.epochs):
                spearman, kendall, classification_metrics, reg_names, class_names = spearman_test(valid_loader, model, opt, device='cuda')

                print(f"Spearman Correlations: {', '.join(f'{x:.3f}' for x in spearman)}")
                print(f"Kendall Correlations: {', '.join(f'{x:.3f}' for x in kendall)}")
                # Log all metrics to wandb
                log_metrics_to_wandb(epoch, spearman, kendall, classification_metrics, reg_names, class_names)

                # Track best model based on average Spearman correlation (only if we have Spearman values)
                if spearman and len(spearman) > 0:
                    avg_spearman = sum(spearman) / len(spearman)
                    if avg_spearman > best_avg_spearman:
                        save_file = os.path.join(opt.save_folder, f'best_encoder.pth')
                        save_dict = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_avg_spearman': best_avg_spearman,
                            'best_epoch': best_epoch,
                        }
                        torch.save(save_dict, save_file)
                        best_model = copy.deepcopy(model)
                        print(f"New best model found! Avg Spearman: {avg_spearman:.3f} at epoch {epoch}")
                        best_avg_spearman = avg_spearman
                        best_epoch = epoch
                        
                if sum(opt.output_dimension_list) == opt.output_dim:
                    plot_latent(valid_loader, model, epoch, opt)
                    # plot_latent(valid_loader, model, epoch, opt, plot_3d=True)
                    
                # Log correlation metrics with proper epoch metric
                corr_log = {"encoder/epoch": epoch}
                for i, (s, k) in enumerate(zip(spearman, kendall)):
                    corr_log[f"encoder/spearman_corr_{i}"] = s
                    corr_log[f"encoder/kendall_corr_{i}"] = k
                
                # Log best model info
                if spearman and len(spearman) > 0:
                    corr_log["encoder/avg_spearman"] = sum(spearman) / len(spearman)
                    corr_log["encoder/best_avg_spearman"] = best_avg_spearman
                    corr_log["encoder/best_epoch"] = best_epoch
                
                wandb.log(corr_log)

                if epoch % opt.save_curr_freq == 0:
                    save_file = os.path.join(opt.save_folder, 'last_encoder.pth')
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(save_dict, save_file)
        
        print("Encoder training completed")
    else:
        print("Skipping encoder training - already completed")
        
    # After finishing encoder training, decide which model to use
    if best_model is not None:
        # Load the best model for regressor training
        model = copy.deepcopy(best_model)
        print(f"Using best model from epoch {best_epoch}")
        
        # disable inference of best model here to save time
        spearman, kendall, classification_metrics, reg_names, class_names = spearman_test(valid_loader, model, opt, device='cuda')
        
        plot_latent(valid_loader, model, best_epoch, opt)
        plot_latent(valid_loader, model, best_epoch, opt, plot_3d=True)
        print(f"Spearman Correlations: {', '.join(f'{x:.3f}' for x in spearman)}")
        print(f"Kendall Correlations: {', '.join(f'{x:.3f}' for x in kendall)}")
        if spearman and len(spearman) > 0:
            best_avg_spearman = sum(spearman) / len(spearman)
            print(f"Best model average Spearman correlation: {best_avg_spearman:.3f}")
    else:
        # If no Spearman tracking (classification only or no valid correlations), use final model
        print("Using final epoch model (no Spearman correlation tracking available)")

    # 3) Freeze encoder for regressor training
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 4) Build the regressor & its optimizers
    # opt.decoder_loss_list = ['l1', 'l1','l1','l1','l1']
    regressor = Regressors(model.feature_dim, opt.output_dim, opt.decoder_loss_list, num_cls=train_loader.dataset.num_classes).cuda()

    # initilaize regressor with eigenvectors if available to speed up training
    if opt.method == "Domain":
        all_proj_coeffs = []
        all_labels = []

        with torch.no_grad():
            for idx, data_tuple in enumerate(train_loader):
                image_0, labels = data_tuple

                # Prepare data: concatenate the two augmented images and move to GPU.
                labels = labels.cuda(non_blocking=True)
                image_0 = torch.cat([image_0[0], image_0[1]], dim=0).cuda(non_blocking=True)

                f_proj, f, proj_coeffs  = model(image_0)
                all_proj_coeffs.append(proj_coeffs.cpu())
                all_labels.append(labels.cpu())

            all_proj_coeffs = torch.cat(all_proj_coeffs, dim=0)  # (dataset, feat_dim)
            all_labels = torch.cat(all_labels, dim=0)  # (dataset, label_dim)
            bsz = all_labels.shape[0]
            all_proj_coeffs = all_proj_coeffs[:bsz]
            all_labels = all_labels.squeeze()

        # Scale and sign
        s  = all_labels.std(0) / all_proj_coeffs.std(0)    # positive scalars
        sign = torch.sign((all_proj_coeffs * all_labels).mean(0))   # +1 or -1
        scale = s * sign                     # final (K,)

        regressor.load_eigen_params(model.prev_evecs, model.running_mean, scale)
    
    # Create optimizers for each output dimension
    reg_optimizers = {
        i: torch.optim.Adam(
            regressor.decoders[i].parameters(),
            lr=opt.learning_rate * 0.1,
            weight_decay=opt.weight_decay * 0.1
        )
        for i in range(opt.output_dim)
    }

    # Create criterion list for regressor based on decoder_loss_list
    criterion_reg_list = []

    print(f"Creating regressor loss functions for outputs: {opt.output_list}")
    print(f"Using regressor loss types: {opt.decoder_loss_list}")

    for i, (output_type, loss_type) in enumerate(zip(opt.output_list, opt.decoder_loss_list)):
        print(f"Regressor Output {i}: {output_type} -> Loss: {loss_type}")
        
        if loss_type.lower() == "l1":
            criterion = torch.nn.L1Loss().cuda()
            print(f" Created L1Loss for regressor {output_type}")
        elif loss_type.lower() == "l2" or loss_type.lower() == "mse":
            criterion = torch.nn.MSELoss().cuda()
            print(f" Created MSELoss for regressor {output_type}")
        elif loss_type.lower() == "crossentropy" or loss_type.lower() == "ce":
            criterion = torch.nn.CrossEntropyLoss().cuda()
            print(f" Created CrossEntropyLoss for regressor {output_type}")
        elif loss_type.lower() == "huber":
            criterion = torch.nn.HuberLoss().cuda()
            print(f" Created HuberLoss for regressor {output_type}")
        else:
            raise ValueError(f"Unknown regressor loss type: {loss_type} for output: {output_type}")
        
        criterion_reg_list.append(criterion)

    print(f"Created {len(criterion_reg_list)} regressor loss functions total")
    
    # 5) Load regressor checkpoint if available
    reg_start_epoch = 1
    if opt.ckpt_regressor and os.path.isfile(opt.ckpt_regressor):
        ckptR = torch.load(opt.ckpt_regressor, map_location='cpu')
        regressor.load_state_dict(ckptR['regressor_state_dict'])
        if 'optimizer_states' in ckptR:
            for i in range(opt.output_dim):
                if i in ckptR['optimizer_states']:
                    reg_optimizers[i].load_state_dict(ckptR['optimizer_states'][i])
        reg_start_epoch = ckptR.get('epoch', 0) + 1
        print(f"Loaded regressor from {opt.ckpt_regressor}, resuming at epoch {reg_start_epoch}")

    # 6) Determine regressor training epochs
    if opt.decoder_epochs is not None:
        final_epoch = opt.decoder_epochs
    else:
        final_epoch = opt.epochs 
        
    print(f"Regressor training will run from epoch {reg_start_epoch} to {final_epoch}.")

    # Skip regressor training if already completed
    if reg_start_epoch > final_epoch:
        print("Regressor training already completed")
    else:
        for epoch in range(reg_start_epoch, final_epoch + 1):
            for i in range(opt.output_dim):
                adjust_learning_rate(opt, reg_optimizers[i], epoch)
            
            # Log regressor learning rates with proper epoch metric
            reg_log = {'regressor/epoch': epoch}
            for i in range(opt.output_dim):
                reg_log[f"regressor/lr_{i}"] = reg_optimizers[i].param_groups[0]['lr']
            
            wandb.log(reg_log)

            print("Training unified regressor")
            measure_train_memory(train_regressors, train_loader, model, regressor, criterion_reg_list,
                            reg_optimizers, epoch, opt)
            
            test(valid_loader, model, regressor, opt, device="cuda", val=True, epoch=epoch)

            # Save regressor checkpoint every save_freq
            if (epoch % opt.save_freq) == 0:
                save_path = os.path.join(opt.save_folder, f'regressor_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'regressor_state_dict': regressor.state_dict(),
                    'optimizer_states': {i: reg_optimizers[i].state_dict()
                                    for i in reg_optimizers}
                }, save_path)
                print(f"Saved regressor checkpoint: {save_path}")

            # Always update the "last" regressor
            last_path = os.path.join(opt.save_folder, 'last_regressor.pth')
            torch.save({
                'epoch': epoch,
                'regressor_state_dict': regressor.state_dict(),
                'optimizer_states': {i: reg_optimizers[i].state_dict()
                                for i in reg_optimizers}
            }, last_path)

        print("Finished regressor training.")
        
    # uncomment to get latent space visualization after training
    # We continue doing visualization. Get the save_freq, and model_path.
    # get all previous checkpoints from encoder only and use plot latent in utils for visualization
    # print("Starting latent space visualization for saved encoder checkpoints")
    # for epoch in range(opt.epochs - opt.save_freq * 3, opt.epochs + 1, opt.save_freq):
    #     encoder_path = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
    #     if os.path.exists(encoder_path):
    #         print(f"Visualizing encoder checkpoint from {encoder_path}")
    #         model.load_state_dict(torch.load(encoder_path, map_location='cuda')['model_state_dict'])
    #         final_spearman, final_kendall, classification_metrics, reg_names, class_names = spearman_test(valid_loader, model, opt, device='cuda')

    #         # Log all metrics to wandb
    #         log_metrics_to_wandb(epoch, final_spearman, final_kendall, classification_metrics, reg_names, class_names, test=True)
    #         # plot_latent(valid_loader, model, epoch, opt)
    #         # plot_latent(valid_loader, model, epoch, opt, plot_3d=True)
    #         plot_3d_latent(valid_loader, model, epoch, opt)
    #     else:
    #         print(f"Encoder checkpoint {encoder_path} does not exist, skipping visualization.")

    # Uncomment to do final evaluation on test set after training for all epochs
    # for epoch in range(15, opt.decoder_epochs + 1, opt.save_freq):
    #     decoder_path = os.path.join(opt.save_folder, f'regressor_epoch_{epoch}.pth')
    #     if os.path.exists(decoder_path):
    #         print(f"Rerun decoder checkpoint from {decoder_path}")
    #         regressor.load_state_dict(torch.load(decoder_path, map_location='cuda')['regressor_state_dict'])
    #         test(test_loader, model, regressor, opt, device="cuda", val=False, epoch=epoch)
    #         final_spearman, final_kendall, classification_metrics, reg_names, class_names = spearman_test(test_loader, model, opt, device='cuda')

    #         # Log all metrics to wandb
    #         log_metrics_to_wandb(epoch, final_spearman, final_kendall, classification_metrics, reg_names, class_names, test=True)
    #     else:
    #         print(f"Encoder checkpoint {decoder_path} does not exist, skipping visualization.")
    
    # 8) Final evaluation on test set
    plot_3d_latent(test_loader, model, 0, opt)
    test(test_loader, model, regressor, opt, device="cuda", val=False)
    
    final_spearman, final_kendall, classification_metrics, reg_names, class_names = spearman_test(test_loader, model, opt, device='cuda')

    print(f"Spearman Correlations: {', '.join(f'{x:.3f}' for x in final_spearman)}")
    print(f"Kendall Correlations: {', '.join(f'{x:.3f}' for x in final_kendall)}")

    # Log all metrics to wandb
    log_metrics_to_wandb(0, final_spearman, final_kendall, classification_metrics, reg_names, class_names, test=True)
    
    # Log final test results with proper epoch - use next sequential epoch
    final_log = {'test/epoch': 1}
    for i, (s, k) in enumerate(zip(final_spearman, final_kendall)):
        final_log[f"test/spearman_corr_{i}"] = s
        final_log[f"test/kendall_corr_{i}"] = k
    
    final_log["test/avg_spearman_corr"] = sum(final_spearman) / len(final_spearman)
    final_log["test/avg_kendall_corr"] = sum(final_kendall) / len(final_kendall)
        
    wandb.log(final_log)
        
    test_retrivial(test_loader, model, regressor, opt, device="cuda")

    print(f"Training and testing finished!")
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()