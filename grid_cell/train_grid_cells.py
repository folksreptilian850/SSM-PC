# train_grid_cells.py

import os
import sys
import time
import shutil
import argparse
import resource
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from config import Config
# from models.toroidal_grid_cell import GridCellNetwork, GridCellSupervisedLoss
from models.toroidal_grid_cell import GridCellNetwork, InitialStateRegularizedLoss
from models.place_hd_cells import PlaceCellEnsemble, HeadDirectionCellEnsemble
from datasets.navigation_dataset import EnhancedNavigationDataset
from utils.visualization import NavigationVisualizer

import torch.multiprocessing as mp
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import contextmanager

# Add this near the top of your train_grid_cells.py file, after the imports

# Worker shutdown handling
import signal
import gc

class SimplifiedLoss(nn.Module):





    def __init__(self, initial_frames_weight=5.0, decay_factor=0.8):
        super().__init__()
        self.initial_frames_weight = initial_frames_weight
        self.decay_factor = decay_factor

    def forward(self, outputs, targets):
        place_logits = outputs['place_logits']  # [B, seq, n_place]
        place_targets = targets['place_targets']  # [B, seq, n_place]
        
        hd_logits = outputs['hd_logits']  # [B, seq, n_hd]
        hd_targets = targets['hd_targets']  # [B, seq, n_hd]
        
        batch_size, seq_len = place_logits.shape[:2]
        
                    
        total_place_loss = 0.0
        total_hd_loss = 0.0
        total_weight = 0.0
        
        for t in range(seq_len):
            if t == 0:
                weight = self.initial_frames_weight * 2.0
            elif t < 5:
                weight = self.initial_frames_weight * (self.decay_factor ** (t-1))
            else:
                weight = 1.0
            
            place_log_softmax_t = F.log_softmax(place_logits[:, t], dim=-1)
            place_loss_t = F.kl_div(
                place_log_softmax_t,
                place_targets[:, t],
                reduction='batchmean'
            )
            
            hd_log_softmax_t = F.log_softmax(hd_logits[:, t], dim=-1)
            hd_loss_t = F.kl_div(
                hd_log_softmax_t,
                hd_targets[:, t],
                reduction='batchmean'
            )
            
            total_place_loss += weight * place_loss_t
            total_hd_loss += weight * hd_loss_t
            total_weight += weight
        
        avg_place_loss = total_place_loss / total_weight
        avg_hd_loss = total_hd_loss / total_weight
        total_loss = avg_place_loss + avg_hd_loss
        
                     
        first_place_pred = F.log_softmax(place_logits[:, 0], dim=-1)
        first_place_target = place_targets[:, 0]
        init_consistency_loss = F.kl_div(
            first_place_pred,
            first_place_target,
            reduction='batchmean'
        ) * 2.0
        
                 
        continuity_loss = 0.0
        if seq_len > 1:
            for t in range(1, min(6, seq_len)):
                continuity_weight = 1.0 - 0.15 * (t-1)
                place_pred_prev = F.softmax(place_logits[:, t-1], dim=-1)
                place_pred_curr = F.softmax(place_logits[:, t], dim=-1)
                
                continuity_loss += continuity_weight * F.kl_div(
                    torch.log(place_pred_curr + 1e-10),
                    place_pred_prev,
                    reduction='batchmean'
                )
        
                                    
        final_loss = total_loss + init_consistency_loss + 0.1 * continuity_loss
        
        return {
            'total': final_loss,
            'place': avg_place_loss.detach().cpu().item(),
            'hd': avg_hd_loss.detach().cpu().item(),
            'init_consistency': init_consistency_loss.detach().cpu().item(),
            'continuity': continuity_loss.detach().cpu().item() if isinstance(continuity_loss, torch.Tensor) else 0.0,
        }


def limit_worker_memory():
    """Limit worker memory usage to prevent OOM errors"""
    try:
        # Set soft limit to 2GB per worker
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # 2GB memory limit per worker
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, hard))  
        
        # Also limit CPU time to prevent infinite loops
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        # 1 hour CPU time limit per worker
        resource.setrlimit(resource.RLIMIT_CPU, (60 * 60, hard))  
        
        # Make sure system calls fail rather than block indefinitely
        # Set timeout for system calls
        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC) 
        # 1024 child processes per worker
        resource.setrlimit(resource.RLIMIT_NPROC, (1024, hard))  
    except Exception as e:
        print(f"Warning: Could not set resource limits: {e}")

# Modify init_worker function to include memory limits
def init_worker():
    """Initialize worker process with resource limits"""
    # Ignore SIGINT in worker processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Set resource limits
    limit_worker_memory()
    
    # Set a more graceful way to handle SIGTERM
    def graceful_exit(signum, frame):
        print(f"Worker process received signal {signum}, exiting gracefully...")
        gc.collect()  # Explicitly run garbage collection
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, graceful_exit)
    signal.signal(signal.SIGABRT, graceful_exit)


def backup_code(source_dir, backup_dir):

    os.makedirs(backup_dir, exist_ok=True)
    dirs_to_backup = ['models', 'utils', 'datasets']
    files_to_backup = ['train_grid_cells.py', 'config.py']
    for dir_name in dirs_to_backup:
        src_dir = os.path.join(source_dir, dir_name)
        dst_dir = os.path.join(backup_dir, dir_name)
        if os.path.exists(src_dir):
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    for file_name in files_to_backup:
        src_file = os.path.join(source_dir, file_name)
        dst_file = os.path.join(backup_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)


class TeeStream:


    def __init__(self, file_stream, terminal_stream):
        self.file_stream = file_stream
        self.terminal_stream = terminal_stream

    def write(self, data):
        self.file_stream.write(data)
        self.terminal_stream.write(data)
        self.terminal_stream.flush()            

    def flush(self):
        self.file_stream.flush()
        self.terminal_stream.flush()


def setup_logger(log_dir):

    log_file = os.path.join(log_dir, "train.log")
    file_stream = open(log_file, "w")

                        
    original_stdout = sys.stdout
    original_stderr = sys.stderr

             
    sys.stdout = TeeStream(file_stream, original_stdout)
    sys.stderr = TeeStream(file_stream, original_stderr)

    print(f"Logging started. All outputs will be saved to {log_file}")


                                 
class SimpleNavigationVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._nature_visualizer = NavigationVisualizer(save_dir)

    def plot_metrics(self, metrics_history, epoch, suffix=""):
        """Delegates to the Nature-style visualizer for consistency."""
        self._nature_visualizer.plot_metrics(metrics_history, epoch, suffix)
    
    def visualize_grid_cells(self, model, dataloader, device, epoch):

        model.eval()
        
                        
        batch = next(iter(dataloader))
        
        positions = batch['positions'].to(device)
        angles = batch['angles'].to(device)
        velocities = batch['velocities'].to(device)
        ang_vels = batch['angular_velocities'].to(device)
        
        with torch.no_grad():
                    
            w = ang_vels.unsqueeze(-1)
            velocity_input = torch.cat([velocities, w], dim=-1)
            init_pos = positions[:,0]
            init_hd = angles[:,0]
            
                    
            outputs = model(velocity_input, init_pos, init_hd)
            
                            
            bottleneck = outputs['bottleneck'][0].reshape(-1, model.module.bottleneck_size if isinstance(model, DDP) else model.bottleneck_size)
            
                            
            corr_matrix = torch.corrcoef(bottleneck.T)
            
                               
            plt.figure(figsize=(15, 5))
            
                       
            plt.subplot(1, 3, 1)
            plt.imshow(corr_matrix.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title('Feature Correlation Matrix')
            
                            
            plt.subplot(1, 3, 2)
            for i in range(min(10, bottleneck.shape[1])):
                plt.plot(bottleneck[:, i].cpu().numpy(), label=f'Neuron {i+1}')
            plt.title('Bottleneck Neuron Activities')
            plt.xlabel('Position Index')
            plt.ylabel('Activation')
            
                                
            plt.subplot(1, 3, 3)
            place_logits = outputs['place_logits'][0]          
            place_probs = torch.softmax(place_logits, dim=-1)
            plt.imshow(place_probs.cpu().numpy(), aspect='auto')
            plt.title('Place Cell Activations')
            plt.xlabel('Place Cell Index')
            plt.ylabel('Time Step')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'grid_cell_viz_epoch_{epoch}.png'))
            plt.close()
            
            return 0.5                                

    def visualize_epoch(self, epoch, model, dataloader, device, metrics_history, suffix=""):

                        
        self.plot_metrics(metrics_history, epoch, suffix)
        grid_score = self.visualize_grid_cells(model, dataloader, device, epoch, suffix)
        return grid_score


class GridCellTrainer:
    def __init__(self, config, gpu_id, world_size, disable_grid_loss=False):
        self.config = config
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.is_main_process = (gpu_id == 0)
        self.device = torch.device(f'cuda:{gpu_id}')
                
        self.disable_grid_loss = disable_grid_loss
        if disable_grid_loss and self.is_main_process:
            print("[info] Grid losses disabled (ablation mode).")

                
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join("log_train_grid_cells", self.timestamp)
        if self.is_main_process:
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_dir, "visualizations"), exist_ok=True)
            setup_logger(self.exp_dir)

            backup_dir = os.path.join(self.exp_dir, "code_backup")
            backup_code(os.path.dirname(os.path.abspath(__file__)), backup_dir)
            print(f"Code backup completed at {backup_dir}")
                          
            # self.visualizer = SimpleNavigationVisualizer(os.path.join(self.exp_dir, "visualizations"))
            self.visualizer = NavigationVisualizer(os.path.join(self.exp_dir, "visualizations")) 

                               
        self.place_cells = PlaceCellEnsemble(
            n_cells=config.PLACE_CELLS_N,
            scale=config.PLACE_CELLS_SCALE,
            pos_min=0,
            pos_max=config.ENV_SIZE,
            seed=config.SEED
        )
        self.hd_cells = HeadDirectionCellEnsemble(
            n_cells=config.HD_CELLS_N,
            concentration=config.HD_CELLS_CONCENTRATION,
            seed=config.SEED
        )

                                                    
        self.model = GridCellNetwork(
            place_cells=self.place_cells,
            hd_cells=self.hd_cells,
            input_size=3,
            hidden_size=config.HIDDEN_SIZE,
            bottleneck_size=256,
            dropout_rate=config.DROPOUT_RATE
        ).to(self.device)

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[gpu_id], find_unused_parameters=True)

              
        # self.criterion = GridCellSupervisedLoss().to(self.device)
        # self.criterion = InitialStateRegularizedLoss(initial_frames_weight=5.0, decay_factor=0.8).to(self.device)
                    
        if disable_grid_loss:
                            
            self.criterion = SimplifiedLoss(initial_frames_weight=5.0, decay_factor=0.8).to(self.device)
        else:
                            
            self.criterion = InitialStateRegularizedLoss(initial_frames_weight=5.0, decay_factor=0.8).to(self.device)


                                                                              
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=config.NATURE_LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.NATURE_WEIGHT_DECAY
        )

                
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

               
        self.dataloaders = self.get_dataloaders()

              
        self.metrics_history = defaultdict(list)

    # Add these changes to the get_dataloaders method in GridCellTrainer class

    def get_dataloaders(self):




        trajectory_folders = [
            d for d in os.listdir(self.config.DATA_ROOT)
            if os.path.isdir(os.path.join(self.config.DATA_ROOT, d))
               and d.startswith('D')
        ]
        trajectory_folders.sort()

        n_total = len(trajectory_folders)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)

        dataset_dirs = {'train': [], 'val': []}
        for folder in trajectory_folders[:n_train]:
            dataset_dirs['train'].append(os.path.join(self.config.DATA_ROOT, folder, '000000'))
        for folder in trajectory_folders[n_train:n_train + n_val]:
            dataset_dirs['val'].append(os.path.join(self.config.DATA_ROOT, folder, '000000'))

        loaders = {}
        for split in ['train', 'val']:
            maze_dirs = dataset_dirs[split]

                           
            chunk_size = 800                     
            dataset = EnhancedNavigationDataset(
                maze_dirs=maze_dirs,
                sequence_length=self.config.SEQUENCE_LENGTH,
                stride=self.config.SEQUENCE_STRIDE,
                split=split,
                chunk_size=chunk_size,
                current_chunk=0           
            )

            if self.world_size > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.gpu_id,
                    shuffle=(split == 'train')
                )
                shuffle_flag = False
            else:
                sampler = None
                shuffle_flag = (split == 'train')

            bs = self.config.BATCH_SIZE if split == 'train' else (
                self.config.BATCH_SIZE // 2 if self.config.BATCH_SIZE > 1 else 1)

                            
            num_workers = min(2, self.config.NUM_WORKERS)

                        
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                shuffle=shuffle_flag,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=False,                     
                drop_last=(split == 'train'),
                timeout=60,
                persistent_workers=(num_workers > 0),
                prefetch_factor=1 if num_workers > 0 else None,             
            )
            loaders[split] = loader

        return loaders

    def train_epoch(self, epoch):
        self.model.train()
        if self.world_size > 1:
            self.dataloaders['train'].sampler.set_epoch(epoch)

        epoch_losses = defaultdict(list)

                
        total_batches = len(self.dataloaders['train'])

                    
                             
        train_loader = tqdm(enumerate(self.dataloaders['train']),
                            desc=f'Train Epoch {epoch + 1}',
                            total=total_batches,          
                            disable=not self.is_main_process,
                            file=sys.stdout,
                            dynamic_ncols=True)

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            positions = batch['positions'].to(self.device)
            angles = batch['angles'].to(self.device)
            velocities = batch['velocities'].to(self.device)
            ang_vels = batch['angular_velocities'].to(self.device)
            w = ang_vels.unsqueeze(-1)
            velocity_input = torch.cat([velocities, w], dim=-1)
            init_pos = positions[:,0]
            init_hd = angles[:,0]

                                    
            place_targets = self.place_cells.compute_activation(positions)
            hd_targets = self.hd_cells.compute_activation(angles)
            targets = {'place_targets': place_targets, 'hd_targets': hd_targets}

                               
            outputs = self.model(velocity_input, init_pos, init_hd)
            
                                 
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict['total']

                        
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.GRAD_CLIP)
            self.optimizer.step()

                         
            if self.is_main_process:
                train_loader.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'batch': f"{batch_idx + 1}/{total_batches}"
                })

            for k, v in loss_dict.items():
                if k == 'total':
                    epoch_losses[k].append(v.item())
                else:
                    epoch_losses[k].append(v)

        return {k: np.mean(v) for k, v in epoch_losses.items()}

    def validate_epoch(self, epoch):
        self.model.eval()
        epoch_losses = defaultdict(list)

                
        total_batches = len(self.dataloaders['val'])

        val_loader = tqdm(enumerate(self.dataloaders['val']),
                          desc=f'Val Epoch {epoch + 1}',
                          total=total_batches,          
                          disable=not self.is_main_process,
                          file=sys.stdout,
                          dynamic_ncols=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloaders['val']):
                positions = batch['positions'].to(self.device)
                angles = batch['angles'].to(self.device)
                velocities = batch['velocities'].to(self.device)
                ang_vels = batch['angular_velocities'].to(self.device)
                w = ang_vels.unsqueeze(-1)
                velocity_input = torch.cat([velocities, w], dim=-1)
                init_pos = positions[:,0]
                init_hd = angles[:,0]

                place_targets = self.place_cells.compute_activation(positions)
                hd_targets = self.hd_cells.compute_activation(angles)
                targets = {'place_targets': place_targets, 'hd_targets': hd_targets}

                outputs = self.model(velocity_input, init_pos, init_hd)
                loss_dict = self.criterion(outputs, targets)

                             
                if self.is_main_process:
                    val_loader.set_postfix({
                        'loss': f"{loss_dict['total'].item():.4f}",
                        'batch': f"{batch_idx + 1}/{total_batches}"
                    })

                for k, v in loss_dict.items():
                    if k == 'total':
                        epoch_losses[k].append(v.item())
                    else:
                        epoch_losses[k].append(v)
        return {k: np.mean(v) for k, v in epoch_losses.items()}

    def train(self):

        if self.is_main_process:
            print("\n" + "=" * 50)
            print("Starting chunked training loop.")
            print("=" * 50 + "\n")

        best_val_loss = float('inf')
        start_epoch = 0
                     
        processed_chunks = set()

                      
        if self.config.RESUME:
            if self.is_main_process:
                print(f"Resuming training from checkpoint: {self.config.RESUME}")
            try:
                checkpoint = torch.load(self.config.RESUME, map_location=self.device)
                if self.world_size > 1:
                    self.model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']

                             
                if 'processed_chunks' in checkpoint and self.config.RESUME_CHUNK_TRAINING:
                    processed_chunks = checkpoint['processed_chunks']
                    if self.is_main_process:
                        print(f"Restored record of processed chunks: {len(processed_chunks)} entries")

                if 'metrics_history' in checkpoint and self.is_main_process:
                    self.metrics_history = checkpoint['metrics_history']
                if self.is_main_process:
                    print(f"Resume successful; continuing from epoch {start_epoch}")
            except Exception as e:
                if self.is_main_process:
                    print(f"Failed to resume training: {e}")
                    print("Starting training from scratch.")

                   
        train_chunks = self.dataloaders['train'].dataset.total_chunks
        val_chunks = self.dataloaders['val'].dataset.total_chunks

        if self.is_main_process:
            print(f"Training data split into {train_chunks} chunks.")
            print(f"Validation data split into {val_chunks} chunks.")

                             
            if self.config.MEMORY_PROFILING_ENABLED:
                initial_memory = self.monitor_memory()
                print(f"Initial memory usage: {initial_memory:.2f} MB")

        try:
            for epoch in range(start_epoch, self.config.NUM_EPOCHS):
                if self.is_main_process:
                    print(f"\nStarting epoch {epoch + 1}/{self.config.NUM_EPOCHS}")

                             
                if self.config.ROTATE_CHUNKS_WITHIN_EPOCH:
                                    
                    chunks_to_process = list(range(train_chunks))
                    if self.config.SHUFFLE_CHUNKS:
                        import random
                        random.shuffle(chunks_to_process)
                else:
                                          
                    remaining_chunks = [i for i in range(train_chunks) if i not in processed_chunks]
                    if not remaining_chunks:                  
                        processed_chunks = set()
                        remaining_chunks = list(range(train_chunks))

                                        
                    chunks_per_epoch = max(1, len(remaining_chunks) // (self.config.NUM_EPOCHS - epoch))
                    chunks_to_process = remaining_chunks[:chunks_per_epoch]

                    if self.is_main_process:
                        print(f"Processing {chunks_per_epoch} training chunks this epoch ({len(remaining_chunks)} remaining)")

                                
                train_metrics_list = []
                for chunk_idx in chunks_to_process:
                    chunk_start_time = time.time()
                    if self.is_main_process:

                              
                    self.dataloaders['train'].dataset.switch_chunk(chunk_idx)

                                       
                    if self.world_size > 1:
                        self.dataloaders['train'].sampler.set_epoch(epoch * train_chunks + chunk_idx)

                           
                    try:
                        train_metrics = self.train_epoch(epoch)
                        train_metrics_list.append(train_metrics)

                                   
                        processed_chunks.add(chunk_idx)
                        
                                                     
                                                                     
                        if self.is_main_process:
                            try:
                                latest_chunk_path = os.path.join(
                                    self.exp_dir, "checkpoints", f"latest_chunk_{chunk_idx + 1}.pth"
                                )
                                torch.save({
                                    'epoch': epoch + 1,
                                    'chunk_idx': chunk_idx + 1,
                                    'model_state_dict':
                                        self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'scheduler_state_dict': self.scheduler.state_dict(),
                                    'metrics_history': self.metrics_history,
                                    'processed_chunks': processed_chunks if self.config.CHECKPOINT_CHUNK_STATE else set()
                                }, latest_chunk_path)
                                print(f"[Chunk {chunk_idx + 1}] latest model saved to {latest_chunk_path}")
                            except Exception as e:
                                print(f"Failed to save latest chunk model: {e}")
                                import traceback
                                traceback.print_exc()

                                            
                        # if self.is_main_process:
                        #     try:
                                                                              
                                                   
                        #         if val_chunks > 0:
                        #             self.dataloaders['val'].dataset.switch_chunk(0)
                        #         grid_score = self.visualizer.visualize_epoch(
                        #             epoch + 1,
                        #             self.model,
                        #             self.dataloaders['val'],
                        #             self.device,
                        #             self.metrics_history,
                                                                                   
                        #         )
                                                                                            
                        #     except Exception as e:
                                                         
                        #         import traceback
                        #         traceback.print_exc()

                              
                        if self.config.FORCE_GC_BETWEEN_CHUNKS:
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()

                        if self.is_main_process and self.config.MEMORY_PROFILING_ENABLED:
                            current_memory = self.monitor_memory()
                            chunk_time = time.time() - chunk_start_time
                            print(
                                f"Chunk {chunk_idx + 1} complete in {chunk_time:.2f}s; memory usage {current_memory:.2f} MB")

                    except Exception as e:
                        if self.is_main_process:
                            print(f"Training chunk {chunk_idx + 1} failed: {e}")
                            import traceback
                            traceback.print_exc()
                            print("Skipping failed chunk and continuing.")

                            
                if train_metrics_list:
                    train_metrics = {k: np.mean([metrics[k] for metrics in train_metrics_list if k in metrics])
                                     for k in train_metrics_list[0].keys()}
                else:
                    train_metrics = {'total': float('inf'), 'place': float('inf'), 'hd': float('inf')}

                           
                val_metrics_list = []

                                     
                val_chunks_to_evaluate = list(range(val_chunks))
                if val_chunks > 5:                     
                    import random
                    val_chunks_to_evaluate = random.sample(val_chunks_to_evaluate, min(5, val_chunks))

                for chunk_idx in val_chunks_to_evaluate:
                    if self.is_main_process:

                              
                    self.dataloaders['val'].dataset.switch_chunk(chunk_idx)

                           
                    try:
                        val_chunk_metrics = self.validate_epoch(epoch)
                        val_metrics_list.append(val_chunk_metrics)

                              
                        if self.config.FORCE_GC_BETWEEN_CHUNKS:
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()

                    except Exception as e:
                        if self.is_main_process:
                            print(f"Validation chunk {chunk_idx + 1} failed: {e}")
                            import traceback
                            traceback.print_exc()

                              
                if val_metrics_list:
                    val_metrics = {k: np.mean([metrics[k] for metrics in val_metrics_list if k in metrics])
                                   for k in val_metrics_list[0].keys()}
                else:
                    val_metrics = {'total': float('inf'), 'place': float('inf'), 'hd': float('inf')}

                            
                if self.is_main_process:
                    try:
                        self.scheduler.step(val_metrics['total'])
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"Updated learning rate: {current_lr:.2e}")
                    except Exception as e:
                        print(f"Failed to update learning rate: {e}")

                          
                    try:
                        for k, v in train_metrics.items():
                            self.metrics_history[f"train_{k}"].append(v)
                        for k, v in val_metrics.items():
                            self.metrics_history[f"val_{k}"].append(v)

                        print(f"\nEpoch {epoch + 1} summary:")
                        print("Train metrics:", {k: f"{v:.4f}" for k, v in train_metrics.items()})
                        print("Validation metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})
                        print(f"Processed chunks: {len(processed_chunks)}/{train_chunks}")
                    except Exception as e:
                        print(f"Failed to log metrics: {e}")

                         
                    try:
                        if (epoch + 1) % self.config.VIZ_INTERVAL == 0:
                            print("\nRunning visualisation...")
                            grid_score = self.visualizer.visualize_epoch(epoch + 1, self.model, self.dataloaders['val'],
                                                                         self.device, self.metrics_history)
                            print(f"Grid score for epoch {epoch + 1}: {grid_score:.3f}")
                    except Exception as e:
                        print(f"Visualisation failed: {e}")
                        import traceback
                        traceback.print_exc()

                          
                    try:
                                
                        if val_metrics['total'] < best_val_loss:
                            best_val_loss = val_metrics['total']
                            save_path = os.path.join(self.exp_dir, "checkpoints", "best_model.pth")
                            print(f"Improved model found; saved to {save_path}")
                            torch.save({
                                'epoch': epoch + 1,
                                'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict(),
                                'best_val_loss': best_val_loss,
                                'metrics_history': self.metrics_history,
                                'processed_chunks': processed_chunks if self.config.CHECKPOINT_CHUNK_STATE else set()
                            }, save_path)

                                 
                        if (epoch + 1) % self.config.SAVE_FREQUENCY == 0:
                            save_path = os.path.join(self.exp_dir, "checkpoints", f"epoch_{epoch + 1}.pth")
                            print(f"Saved periodic checkpoint to {save_path}")
                            torch.save({
                                'epoch': epoch + 1,
                                'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict(),
                                'metrics_history': self.metrics_history,
                                'processed_chunks': processed_chunks if self.config.CHECKPOINT_CHUNK_STATE else set()
                            }, save_path)
                    except Exception as e:
                        print(f"Failed to save model: {e}")
                        import traceback
                        traceback.print_exc()

                            
                    if self.config.MEMORY_PROFILING_ENABLED:
                        current_memory = self.monitor_memory()
                        print(f"End of epoch {epoch + 1}: memory usage {current_memory:.2f} MB")

                         
                if self.world_size > 1:
                    try:
                        dist.barrier()
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Distributed barrier failed: {e}")

        except KeyboardInterrupt:
            if self.is_main_process:
                print("\nKeyboard interrupt received; saving checkpoint and shutting down.")
                try:
                    save_path = os.path.join(self.exp_dir, "checkpoints", "interrupted.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'metrics_history': self.metrics_history,
                        'processed_chunks': processed_chunks if self.config.CHECKPOINT_CHUNK_STATE else set()
                    }, save_path)
                    print(f"Interrupt checkpoint saved to {save_path}")
                except Exception as e:
                    print(f"Failed to save interrupt checkpoint: {e}")

        except Exception as e:
            if self.is_main_process:
                print(f"Unhandled exception in training loop: {e}")
                import traceback
                traceback.print_exc()
                try:
                    save_path = os.path.join(self.exp_dir, "checkpoints", "error_recovery.pth")
                    torch.save({
                        'epoch': epoch + 1 if 'epoch' in locals() else start_epoch,
                        'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'metrics_history': self.metrics_history,
                        'processed_chunks': processed_chunks if self.config.CHECKPOINT_CHUNK_STATE else set()
                    }, save_path)
                    print(f"Recovery checkpoint saved to {save_path}")
                except Exception as e2:
                    print(f"Failed to save recovery checkpoint: {e2}")

        finally:
            if self.is_main_process:
                print("\n" + "=" * 50)
                print("Training complete.")
                print("=" * 50)

                          
                if self.config.MEMORY_PROFILING_ENABLED:
                    final_memory = self.monitor_memory()
                    print(f"Final memory usage: {final_memory:.2f} MB")


    def manage_memory(threshold_mb=10000):

        memory_mb = self.monitor_memory()
        if memory_mb > threshold_mb:
            print(f"Memory usage {memory_mb:.2f} MB exceeds threshold; running cleanup.")
            torch.cuda.empty_cache()
            import gc
            gc.collect()

                      
    def monitor_memory(self):

        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # MB
            return memory_mb
        except ImportError:
            return -1            

def train_worker(gpu_id, world_size, args, config):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=gpu_id
        )
        torch.cuda.set_device(gpu_id)
                                 
    trainer = GridCellTrainer(config, gpu_id, world_size, disable_grid_loss=args.disable_grid_loss)
    
    try:
        trainer.train()
    except Exception as e:
        print(f"[rank={gpu_id}] Error occurred: {str(e)}")
        raise e
    finally:
        if world_size > 1:
            dist.destroy_process_group()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs (comma-separated)')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path')
    parser.add_argument('--disable_grid_loss', action='store_true', help='Disable grid-related loss terms (ablation).')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_ids = [i for i in range(len(args.gpus.split(',')))]
    world_size = len(gpu_ids)

    config = Config()                                                      
    if args.resume:
        config.RESUME = args.resume

    if world_size > 1:
        mp.spawn(
            train_worker,
            args=(world_size, args, config),
            nprocs=world_size,
            join=True
        )
    else:
        train_worker(0, world_size, args, config)

if __name__ == "__main__":
    main()
