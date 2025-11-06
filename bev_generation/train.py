# /bev_predictor/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import numpy as np

from config import Config
from models.bev_net import BEVPredictor
from models.losses import BEVLoss
from datasets.maze_dataset import get_dataloaders

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

def train_model(config):
          
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
            
    os.makedirs(config.INFERENCE_DIR, exist_ok=True)
    
             
    dataloaders = get_dataloaders(config)
    print(f"Training samples: {len(dataloaders['train'].dataset)}")
    print(f"Validation samples: {len(dataloaders['val'].dataset)}")
    
          
    model = BEVPredictor(config).to(device)
    
            
    criterion = BEVLoss().to(device)
    
         
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
            
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
              
        model.train()
        train_losses = []
        
        for batch in tqdm(dataloaders['train'], desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} - Training'):
            fps_view = batch['fps_view'].to(device)
            bev_view = batch['bev_view'].to(device)
            angle = batch['angle'].to(device)
            
            optimizer.zero_grad()
            pred_bev = model(fps_view, angle)
            loss, loss_components = criterion(pred_bev, bev_view)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss = sum(train_losses) / len(train_losses)
        
              
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloaders['val'], 
                                                 desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} - Validation')):
                fps_view = batch['fps_view'].to(device)
                bev_view = batch['bev_view'].to(device)
                angle = batch['angle'].to(device)
                
                pred_bev = model(fps_view, angle)
                loss, _ = criterion(pred_bev, bev_view)
                val_losses.append(loss.item())
                
                           
                save_inference_results(
                    fps_view, bev_view, pred_bev,
                    batch_idx, epoch + 1,
                    config.INFERENCE_DIR
                )
        
        val_loss = sum(val_losses) / len(val_losses)
        
                      
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
               
        scheduler.step(val_loss)
        
                
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving best model with validation loss: {val_loss:.4f}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
        
                 
        if (epoch + 1) % config.SAVE_FREQUENCY == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    config = Config()
    train_model(config)
