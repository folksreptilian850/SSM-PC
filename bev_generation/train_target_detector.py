# /bev_predictor/train_target_detector.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
from sklearn.metrics import average_precision_score

from config import Config
from models.target_detector import TargetDetector
from models.loss_functions import TargetDetectionLoss
from datasets.maze_dataset import get_dataloaders


                                
import matplotlib.pyplot as plt

def save_code_backup(log_dir):

    backup_dir = os.path.join(log_dir, "code_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
               
    files_to_backup = [
        ('config.py', 'config.py'),
        ('models/loss_functions.py', 'models/loss_functions.py'),
        ('models/target_detector.py', 'models/target_detector.py'),
        ('datasets/maze_dataset.py', 'datasets/maze_dataset.py'),
        ('train_target_detector.py', 'train_target_detector.py')
    ]
    
          
    for src, dst in files_to_backup:
        dst_path = os.path.join(backup_dir, dst)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            import shutil
            shutil.copy2(src, dst_path)
            print(f"Backed up {src} to {dst_path}")
        except Exception as e:
            print(f"Failed to backup {src}: {str(e)}")

class LossTracker:

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {'visibility_ap': [], 'position_error': []}
        self.val_metrics = {'visibility_ap': [], 'position_error': []}
        self.epochs = []
        
    def update(self, epoch, train_loss, val_loss, train_metric, val_metric):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        for k in train_metric:
            if k not in self.train_metrics:
                self.train_metrics[k] = []
            self.train_metrics[k].append(train_metric[k])
            
        for k in val_metric:
            if k not in self.val_metrics:
                self.val_metrics[k] = []
            self.val_metrics[k].append(val_metric[k])
    
    def plot_losses(self):

        plt.figure(figsize=(12, 8))
        
               
        plt.subplot(2, 1, 1)
        plt.plot(self.epochs, self.train_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
              
        plt.subplot(2, 1, 2)
        for metric in self.train_metrics:
            plt.plot(self.epochs, self.train_metrics[metric], 
                    label=f'Train {metric}', linestyle='--')
            plt.plot(self.epochs, self.val_metrics[metric], 
                    label=f'Val {metric}', linestyle='-')
        
        plt.title('Training and Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.log_dir, 'loss_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def visualize_predictions(model, dataloader, device, epoch, log_dir, num_samples=10):











             
    vis_dir = os.path.join(log_dir, f'predictions_epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
             
    model.eval()
    
               
    processed_samples = {
        'no_target': 0,
        'with_target': 0
    }
    
    with torch.no_grad():
        for batch in dataloader:
                        
            fps_view = batch['fps_view'].to(device)
            angle = batch['angle'].to(device)
            gt_visibility = batch['object_visibility']
            gt_positions = batch['object_relative_positions']
            
                  
            outputs = model(fps_view, angle)
            
                       
            for i in range(len(fps_view)):
                             
                has_target = torch.any(gt_visibility[i] > 0.5)
                
                           
                if not has_target and processed_samples['no_target'] < 3:
                    sample_type = 'no_target'
                elif has_target and processed_samples['with_target'] < 7:
                    sample_type = 'with_target'
                else:
                    continue
                
                        
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                plt.subplots_adjust(wspace=0.3)             
                
                         
                fps_img = fps_view[i].cpu().permute(1, 2, 0).numpy()
                fps_img = fps_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                fps_img = np.clip(fps_img, 0, 1)
                axes[0].imshow(fps_img)
                axes[0].set_title('First Person View')
                axes[0].axis('off')
                
                         
                axes[1].set_title('Predictions vs Ground Truth')
                
                            
                pred_visibility = outputs['object_visibility'][i].cpu().numpy()
                gt_vis = gt_visibility[i].cpu().numpy()
                
                          
                x = np.arange(len(pred_visibility))
                axes[1].bar(x - 0.2, pred_visibility, 0.4, label='Predicted', color='blue', alpha=0.6)
                axes[1].bar(x + 0.2, gt_vis, 0.4, label='Ground Truth', color='red', alpha=0.6)
                axes[1].set_xlabel('Object Index')
                axes[1].set_ylabel('Visibility Probability')
                axes[1].set_title('Visibility Prediction')
                axes[1].set_ylim(0, 1)          
                axes[1].legend()
                
                                
                pred_positions = outputs['object_relative_positions'][i].cpu().numpy()
                gt_positions_np = gt_positions[i].cpu().numpy()
                
                          
                info_text = "Relative Positions:\n"
                info_text += "Obj | Pred Vis | Pred Pos  | GT Vis | GT Pos\n"
                info_text += "-" * 50 + "\n"
                for j in range(len(pred_visibility)):
                    info_text += (
                        f"{j:3d} | {pred_visibility[j]:8.3f} | "
                        f"({pred_positions[j][0]:6.3f}, {pred_positions[j][1]:6.3f}) | "
                        f"{gt_vis[j]:6.3f} | "
                        f"({gt_positions_np[j][0]:6.3f}, {gt_positions_np[j][1]:6.3f})\n"
                    )
                
                axes[1].text(
                    1.05, 0.5, info_text, 
                    transform=axes[1].transAxes, 
                    fontfamily='monospace', 
                    verticalalignment='center',
                    fontsize=8
                )
                
                plt.tight_layout()
                
                      
                save_filename = f'epoch_{epoch}_{"no_target" if sample_type == "no_target" else "with_target"}_{processed_samples[sample_type]}.png'
                save_path = os.path.join(vis_dir, save_filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                           
                processed_samples[sample_type] += 1
                
                                       
                if processed_samples['no_target'] >= 3 and processed_samples['with_target'] >= 7:
                    return

                                    
        if processed_samples['no_target'] < 3 or processed_samples['with_target'] < 7:
            print(f"Warning: Could not find enough samples. "
                  f"No target samples: {processed_samples['no_target']}, "
                  f"With target samples: {processed_samples['with_target']}")

def setup_distributed(gpu_id, world_size):
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=gpu_id
        )

def compute_metrics(pred_visibility, pred_positions, true_visibility, true_positions):

    metrics = {}
    
                      
    true_visibility = true_visibility.detach().cpu()
    pred_visibility = pred_visibility.detach().cpu()
    true_positions = true_positions.detach().cpu()
    pred_positions = pred_positions.detach().cpu()
    
              
    true_vis_np = true_visibility.numpy().flatten()
    pred_vis_np = pred_visibility.numpy().flatten()
    
    if len(set(true_vis_np)) > 1:
        metrics['visibility_ap'] = average_precision_score(true_vis_np, pred_vis_np)
    else:
        metrics['visibility_ap'] = 0.0
    
                       
    visible_mask = true_visibility > 0.5
    if visible_mask.sum() > 0:
        position_error = torch.norm(
            pred_positions[visible_mask] - true_positions[visible_mask],
            dim=-1
        ).mean().item()
        metrics['position_error'] = position_error
    else:
        metrics['position_error'] = 0.0
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_val_loss, 
                   is_best, checkpoint_dir, world_size):
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss
    }
    
                    
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(state, latest_path)
    
            
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)
        
                    
    if (epoch + 1) % 5 == 0:
        epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(state, epoch_path)

def train_model(gpu_id, world_size, args, config):
             
    setup_distributed(gpu_id, world_size)
    device = torch.device(f'cuda:{gpu_id}')
    is_main_process = gpu_id == 0
    
                       
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("log_train_target", timestamp)
    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
                 
        log_file = os.path.join(log_dir, 'train.log')
        logger = open(log_file, 'w')

               
        save_code_backup(log_dir)
                 
        loss_tracker = LossTracker(log_dir)
        
        def log_print(*args, **kwargs):

            print(*args, **kwargs)
            print(*args, **kwargs, file=logger)
            logger.flush()            
    
    
    config.CHECKPOINT_DIR = os.path.join(log_dir, "checkpoints")
    
    if world_size > 1:
        dist.barrier()
    
             
    dataloaders = get_dataloaders(config, world_size > 1, gpu_id, world_size)
    
    if is_main_process:
        print(f"Training samples: {len(dataloaders['train'].dataset)}")
        print(f"Validation samples: {len(dataloaders['val'].dataset)}")
    
                   
    model = TargetDetector(config.MAX_OBJECTS).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    
    criterion = TargetDetectionLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
                        
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        if is_main_process:
            print(f"Resumed from epoch {start_epoch}")
    
          
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        if world_size > 1:
            dataloaders['train'].sampler.set_epoch(epoch)
        
              
        model.train()
        train_losses = []
        train_metrics = {'visibility_ap': [], 'position_error': []}
        
        for batch in tqdm(dataloaders['train'], 
                         desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Training",
                         disable=not is_main_process):
                  
            fps_view = batch['fps_view'].to(device)
            angle = batch['angle'].to(device)
            
                  
            optimizer.zero_grad()
            outputs = model(fps_view, angle)
            
                  
            targets = {
                'object_visibility': batch['object_visibility'].to(device),
                'object_relative_positions': batch['object_relative_positions'].to(device),
            }
            loss, loss_components = criterion(outputs, targets)
            
                  
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
                  
            metrics = compute_metrics(
                outputs['object_visibility'],
                outputs['object_relative_positions'],
                targets['object_visibility'],
                targets['object_relative_positions']
            )
            for k, v in metrics.items():
                train_metrics[k].append(v)
        
              
        model.eval()
        val_losses = []
        val_metrics = {'visibility_ap': [], 'position_error': []}
        
        with torch.no_grad():
            for batch in tqdm(dataloaders['val'], desc="Validation", 
                            disable=not is_main_process):
                fps_view = batch['fps_view'].to(device)
                angle = batch['angle'].to(device)
                
                outputs = model(fps_view, angle)
                targets = {
                    'object_visibility': batch['object_visibility'].to(device),
                    'object_relative_positions': batch['object_relative_positions'].to(device),
                }
                
                loss, _ = criterion(outputs, targets)
                val_losses.append(loss.item())
                
                metrics = compute_metrics(
                    outputs['object_visibility'],
                    outputs['object_relative_positions'],
                    targets['object_visibility'],
                    targets['object_relative_positions']
                )
                for k, v in metrics.items():
                    val_metrics[k].append(v)
        
                   
        val_loss = sum(val_losses) / len(val_losses)
        for k in val_metrics:
            val_metrics[k] = sum(val_metrics[k]) / len(val_metrics[k])
        
               
        scheduler.step(val_loss)
        
                           
        if is_main_process:
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                val_metrics, best_val_loss, is_best,
                config.CHECKPOINT_DIR,
                world_size
            )

                                
            visualize_predictions(
                model, 
                dataloaders['val'], 
                device, 
                epoch + 1, 
                log_dir, 
                num_samples=10
            )
            
                    
            log_print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
            log_print(f"Train Loss: {sum(train_losses)/len(train_losses):.4f}")
            log_print(f"Val Loss: {val_loss:.4f}")
            log_print("\nTraining Metrics:")
            for k, v in train_metrics.items():
                log_print(f"{k}: {sum(v)/len(v):.4f}")
            log_print("\nValidation Metrics:")
            for k, v in val_metrics.items():
                log_print(f"{k}: {v:.4f}")
            log_print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

                     
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_metrics = {k: sum(v)/len(v) for k, v in train_metrics.items()}
            
            loss_tracker.update(
                epoch + 1,
                avg_train_loss,
                val_loss,
                avg_train_metrics,
                val_metrics  
            )
            
                       
            loss_tracker.plot_losses()
        
        if world_size > 1:
            dist.barrier()
            
                 
    if is_main_process:
        log_print("\nTraining completed!")
        log_print(f"Best validation loss: {best_val_loss:.4f}")
        logger.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0',
                       help='GPU IDs to use (comma-separated)')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    gpu_ids = [int(id) for id in args.gpus.split(',')]
    world_size = len(gpu_ids)
    
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        mp.spawn(train_model,
                args=(world_size, args, Config()),
                nprocs=world_size,
                join=True)
    else:
        train_model(gpu_ids[0], world_size, args, Config())

if __name__ == '__main__':
    main()
