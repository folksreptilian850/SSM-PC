import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image
import numpy as np
import time, shutil
import sys

from config import Config
from models.bev_net import BEVPredictor
from models.losses import BEVLoss
from datasets.maze_dataset import get_dataloaders

def setup_distributed(gpu_id, world_size):
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=gpu_id
        )

def save_inference_results(fps_view, true_bev, pred_bev, batch_idx, epoch, save_dir):

    epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)

    batch_size = fps_view.size(0)
    for i in range(batch_size):
        sample_dir = os.path.join(epoch_dir, f'batch_{batch_idx}_sample_{i}')
        os.makedirs(sample_dir, exist_ok=True)

                  
        fps = fps_view[i].cpu().permute(1, 2, 0).numpy()
        fps = (fps * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        fps = Image.fromarray(fps.astype(np.uint8))
        fps.save(os.path.join(sample_dir, 'fps_view.png'))

                 
        true = true_bev[i].cpu().permute(1, 2, 0).numpy() * 255
        true = Image.fromarray(true.astype(np.uint8), 'RGBA')
        true.save(os.path.join(sample_dir, 'true_bev.png'))

                 
        pred = pred_bev[i].cpu().permute(1, 2, 0).numpy() * 255
        pred = Image.fromarray(pred.astype(np.uint8), 'RGBA')
        pred.save(os.path.join(sample_dir, 'pred_bev.png'))

def train_model(gpu_id, world_size, args, config):
    setup_distributed(gpu_id, world_size)
    device = torch.device(f'cuda:{gpu_id}')
    is_main_process = gpu_id == 0

            
    root_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(root_dir, "log_train", timestamp)

    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "inference_output"), exist_ok=True)

                
        src_files = [
            os.path.join(root_dir, f) for f in [
                'train_multi_gpu.py', 'config.py',
                'models/bev_net.py', 'models/losses.py',
                'datasets/maze_dataset.py'
            ]
        ]
        for src in src_files:
            if os.path.exists(src):
                dst = os.path.join(log_dir, os.path.relpath(src, root_dir))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

        log_file = open(os.path.join(log_dir, 'training.log'), 'w')
        sys.stdout = log_file

    config.CHECKPOINT_DIR = os.path.join(log_dir, "checkpoints")
    config.INFERENCE_DIR = os.path.join(log_dir, "inference_output")

    if world_size > 1:
        dist.barrier()

              
    model = BEVPredictor(config).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[gpu_id])

    criterion = BEVLoss().to(device)
    
              
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

             
    dataloaders = get_dataloaders(config, world_size > 1, gpu_id, world_size)

    if is_main_process:
        print(f"Training samples: {len(dataloaders['train'].dataset)}")
        print(f"Validation samples: {len(dataloaders['val'].dataset)}")

                  
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if world_size > 1:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            print(f"Loaded checkpoint from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")

          
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        if world_size > 1:
            dataloaders['train'].sampler.set_epoch(epoch)

              
        model.train()
        train_losses = []

        for batch in tqdm(dataloaders['train'],
                         desc=f'Epoch {epoch + 1}/{config.NUM_EPOCHS} - Training',
                         disable=not is_main_process):

            fps_view = batch['fps_view'].to(device)
            bev_view = batch['bev_view'].to(device)
            angle = batch['angle'].to(device)

            optimizer.zero_grad()
            outputs = model(fps_view, angle)
            
            loss, loss_components = criterion(outputs, {'bev_view': bev_view})
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

              
        model.eval()
        val_losses = []

        with torch.no_grad():
            all_samples = []
            for batch_idx, batch in enumerate(tqdm(dataloaders['val'], 
                                                 desc=f"Epoch {epoch + 1} - Validation")):
                fps_view = batch['fps_view'].to(device)
                bev_view = batch['bev_view'].to(device)
                angle = batch['angle'].to(device)

                outputs = model(fps_view, angle)
                loss, _ = criterion(outputs, {'bev_view': bev_view})
                val_losses.append(loss.item())

                             
                for i in range(fps_view.shape[0]):
                    sample_info = {
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'fps_view': batch['fps_view'][i],
                        'bev_view': batch['bev_view'][i],
                        'bev_pred': outputs['bev'][i]
                    }
                    all_samples.append(sample_info)

                     
            if is_main_process and config.SAVE_VISUALIZATIONS and len(all_samples) > 0:
                selected_samples = random.sample(all_samples, min(10, len(all_samples)))
                for idx, sample in enumerate(selected_samples):
                    save_dir = os.path.join(
                        config.INFERENCE_DIR,
                        f'epoch_{epoch + 1}_sample_{idx}'
                    )
                    os.makedirs(save_dir, exist_ok=True)

                          
                    fps = sample['fps_view'].cpu().permute(1, 2, 0).numpy()
                    fps = (fps * np.array([0.229, 0.224, 0.225]) + 
                          np.array([0.485, 0.456, 0.406])) * 255
                    fps = Image.fromarray(fps.astype(np.uint8))
                    fps.save(os.path.join(save_dir, 'fps_view.png'))

                    true = sample['bev_view'].cpu().permute(1, 2, 0).numpy() * 255
                    true = Image.fromarray(true.astype(np.uint8), 'RGBA')
                    true.save(os.path.join(save_dir, 'true_bev.png'))

                    pred = sample['bev_pred'].cpu().permute(1, 2, 0).numpy() * 255
                    pred = Image.fromarray(pred.astype(np.uint8), 'RGBA')
                    pred.save(os.path.join(save_dir, 'pred_bev.png'))

                   
        if world_size > 1:
            val_loss_tensor = torch.tensor(sum(val_losses) / len(val_losses)).to(device)
            dist.all_reduce(val_loss_tensor)
            val_loss = val_loss_tensor.item() / world_size
        else:
            val_loss = sum(val_losses) / len(val_losses)

        if is_main_process:
            print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}:")
            print(f"Training Loss: {sum(train_losses) / len(train_losses):.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

                    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving best model with validation loss: {val_loss:.4f}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))

                            
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pth'))

                            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, os.path.join(config.CHECKPOINT_DIR, 'latest_checkpoint.pth'))

        scheduler.step(val_loss)

        if world_size > 1:
            dist.barrier()

                 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_main_process:
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0',
                      help='GPU IDs to use (comma-separated, e.g., "0,1,2,3")')
    parser.add_argument('--resume', action='store_true',
                      help='resume training from latest checkpoint')
    args = parser.parse_args()

    gpu_ids = [int(id) for id in args.gpus.split(',')]
    world_size = len(gpu_ids)

    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        import torch.multiprocessing as mp
        mp.spawn(train_model,
                args=(world_size, args, Config()),
                nprocs=world_size,
                join=True)
    else:
        train_model(gpu_ids[0], world_size, args, Config())

if __name__ == '__main__':
    main()
