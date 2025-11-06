# /bev_predictor/datasets/maze_dataset.py

import argparse
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from utils.position_utils import PositionConverter
from pathlib import Path

# /bev_predictor/datasets/maze_dataset.py

class SingleMazeDataset(Dataset):
    def __init__(self, maze_dir, max_objects=6):
        self.maze_dir = maze_dir
        self.max_objects = max_objects
        

        self.fps_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.bev_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        

        with open(os.path.join(maze_dir, 'frame_info.json'), 'r') as f:
            self.frame_info_raw = json.load(f)
            

        with open(os.path.join(maze_dir, 'env_config.json'), 'r') as f:
            self.env_config = json.load(f)
            

        self.objects = self.env_config['objects'][:max_objects]
        self.maze_size = self.env_config['maze_size']
        num_objects = len(self.objects)
        

        self.object_colors = torch.zeros((self.max_objects, 3), dtype=torch.float32)
        self.object_positions = torch.zeros((self.max_objects, 2), dtype=torch.float32)
        

        for i, obj in enumerate(self.objects):
            self.object_colors[i] = torch.tensor(obj['color'], dtype=torch.float32)
            self.object_positions[i] = torch.tensor(obj['position'], dtype=torch.float32)
            

        self.fps_dir = os.path.join(maze_dir, 'fps_views')
        self.bev_dir = os.path.join(maze_dir, 'bev_views')
        self.frames = []
        for frame_id in sorted(self.frame_info_raw.keys()):
            normalized_info = self._normalize_frame_info(frame_id, self.frame_info_raw[frame_id])

            fps_rel = normalized_info.get('fps_view_path') or os.path.join('fps_views', f'{frame_id}.png')
            bev_rel = normalized_info.get('bev_view_path') or os.path.join('bev_views', f'{frame_id}.png')

            fps_path = os.path.join(self.maze_dir, fps_rel)
            bev_path = os.path.join(self.maze_dir, bev_rel)

            if not (os.path.exists(fps_path) and os.path.exists(bev_path)):
                continue

            self.frames.append({
                'frame_id': frame_id,
                'fps_path': fps_path,
                'bev_path': bev_path,
                'info': normalized_info
            })

        self.position_converter = PositionConverter(maze_size=self.maze_size)

    def _normalize_frame_info(self, frame_id, frame_info):

        normalized = {
            'frame_id': frame_id
        }

        if 'rotation_angle' in frame_info:
            rotation_angle = frame_info['rotation_angle']
            agent_position = frame_info.get('agent_position', [0.0, 0.0])
            visible_indices = frame_info.get('visible_indices', [])
        elif 'agent_state' in frame_info:
            agent_state = frame_info.get('agent_state', {})
            observations = frame_info.get('observations', {})

            rotation_angle = agent_state.get('rotation')
            agent_position = agent_state.get('position', [0.0, 0.0])
            visible_indices = observations.get('visible_targets', [])

            fps_path = observations.get('fps_view')
            bev_path = observations.get('bev_view')
            if fps_path:
                normalized['fps_view_path'] = fps_path
            if bev_path:
                normalized['bev_view_path'] = bev_path
        else:
            raise KeyError(f'Unsupported frame_info format for frame {frame_id}')

        if rotation_angle is None:
            raise KeyError(f'Missing rotation information for frame {frame_id}')

        if agent_position is None or len(agent_position) != 2:
            raise KeyError(f'Invalid agent position for frame {frame_id}: {agent_position}')

        normalized['rotation_angle'] = float(rotation_angle)
        normalized['agent_position'] = [float(agent_position[0]), float(agent_position[1])]
        normalized['visible_indices'] = [int(idx) for idx in visible_indices] if visible_indices else []

        return normalized

    def _create_object_targets(self, frame_info):

        visibility = torch.zeros(self.max_objects, dtype=torch.float32)
        visible_indices = frame_info['visible_indices']
        
        for idx in visible_indices:
            if idx >= 0 and idx < self.max_objects:
                visibility[idx] = 1.0
        
        agent_pos = torch.tensor(frame_info['agent_position'], dtype=torch.float32)
        agent_angle = float(frame_info['rotation_angle'])
        
        relative_positions = torch.zeros((self.max_objects, 2), dtype=torch.float32)
        for i in range(min(len(self.objects), self.max_objects)):
            if visibility[i] > 0:
                obj_pos = self.object_positions[i]
                

                rel_pos = self.position_converter.absolute_to_relative(
                    obj_pos, agent_pos, agent_angle
                )
                
                relative_positions[i] = torch.tensor(rel_pos, dtype=torch.float32)
        
        return visibility, relative_positions

    def _rotate_bev(self, bev_img, angle_rad):

        angle_deg = np.degrees(angle_rad) + 90
        return bev_img.rotate(angle_deg,
                            resample=Image.BICUBIC,
                            expand=False,
                            center=(bev_img.size[0] / 2, bev_img.size[1] / 2))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame_info = frame['info']
        

        fps_path = frame['fps_path']
        with Image.open(fps_path) as fps_img:
            fps_img = fps_img.convert('RGB')
            fps_tensor = self.fps_transform(fps_img)
        

        bev_path = frame['bev_path']
        with Image.open(bev_path) as bev_img:
            bev_img = bev_img.convert('RGBA')
            rotation_angle = frame_info['rotation_angle']
            rotated_bev = self._rotate_bev(bev_img, rotation_angle)
            bev_tensor = self.bev_transform(rotated_bev)

        

        angle_tensor = torch.tensor([
            np.cos(rotation_angle),
            np.sin(rotation_angle)
        ], dtype=torch.float32)
        

        visibility, relative_positions = self._create_object_targets(frame_info)
        
        return {
            'fps_view': fps_tensor,
            'bev_view': bev_tensor,
            'angle': angle_tensor,
            'object_visibility': visibility,
            'object_relative_positions': relative_positions,
            'object_colors': self.object_colors,
            'object_absolute_positions': self.object_positions.clone(),
            'agent_position': torch.tensor(frame_info['agent_position'], dtype=torch.float32),
            'frame_id': frame['frame_id'],
            'maze_id': os.path.basename(self.maze_dir),
            'rotation_angle': rotation_angle
        }



def visualize_rotation(dataset, idx, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    sample = dataset[idx]


    fig, axes = plt.subplots(1, 2, figsize=(15, 6))


    fps_img = sample['fps_view'].permute(1, 2, 0).numpy()
    fps_img = fps_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    fps_img = np.clip(fps_img, 0, 1)
    axes[0].imshow(fps_img)
    axes[0].set_title('First Person View')
    axes[0].axis('off')


    bev_img = sample['bev_view'].permute(1, 2, 0).numpy()
    axes[1].imshow(bev_img)


    center = np.array([bev_img.shape[1] / 2, bev_img.shape[0] / 2])
    axes[1].arrow(center[0], center[1],
                  0, -50,
                  color='red', width=2, head_width=10)

    angle_deg = np.degrees(sample['rotation_angle'])
    axes[1].set_title(f'BEV View (Agent facing up)\nOriginal Angle: {angle_deg:.1f}°')
    axes[1].axis('off')

    plt.tight_layout()


    save_path = os.path.join(save_dir, f'frame_{sample["frame_id"]}_viz.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Frame {sample['frame_id']}: Original Angle = {angle_deg:.1f}°, Rotated = {angle_deg + 90:.1f}°")

def visualize_rotation_with_targets(dataset, idx, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    sample = dataset[idx]
    maze_size = 15
    img_size = 250
    

    center_x = img_size // 2
    center_y = img_size // 2


    visibility = sample['object_visibility']
    object_colors = sample['object_colors']
    absolute_positions = sample['object_absolute_positions']
    relative_positions = sample['object_relative_positions']
    agent_pos = sample['agent_position'].numpy()
    rotation_angle = sample['rotation_angle']


    fig, axes = plt.subplots(1, 3, figsize=(20, 6))


    fps_img = sample['fps_view'].permute(1, 2, 0).numpy()
    fps_img = fps_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    fps_img = np.clip(fps_img, 0, 1)
    axes[0].imshow(fps_img)
    axes[0].set_title('First Person View')
    axes[0].axis('off')


    bev_img = sample['bev_view'].permute(1, 2, 0).numpy()
    axes[1].imshow(bev_img)
    

    agent_pixel_x = int(agent_pos[0] / maze_size * img_size)
    agent_pixel_y = int((maze_size - agent_pos[1]) / maze_size * img_size)
    

    triangle_size = 10
    dx = np.cos(rotation_angle) * triangle_size
    dy = -np.sin(rotation_angle) * triangle_size
    
    triangle = plt.Polygon([
        [agent_pixel_x + dx, agent_pixel_y + dy],
        [agent_pixel_x + np.cos(rotation_angle + 2.356194) * triangle_size,
         agent_pixel_y - np.sin(rotation_angle + 2.356194) * triangle_size],
        [agent_pixel_x + np.cos(rotation_angle - 2.356194) * triangle_size,
         agent_pixel_y - np.sin(rotation_angle - 2.356194) * triangle_size]
    ], color='red')
    axes[1].add_patch(triangle)
    

    for i in range(len(visibility)):
        if visibility[i] > 0.5:
            obj_pos = absolute_positions[i].numpy()
            obj_pixel_x = int(obj_pos[0] / maze_size * img_size)
            obj_pixel_y = int((maze_size - obj_pos[1]) / maze_size * img_size)
            
            color = object_colors[i].numpy()
            circle = plt.Circle((obj_pixel_x, obj_pixel_y), 5, color=color)
            axes[1].add_patch(circle)
    
    axes[1].set_title(f'Original BEV\nRotation: {np.degrees(rotation_angle):.1f}°')
    axes[1].axis('off')


    axes[2].imshow(bev_img)
    

    triangle = plt.Polygon([
        [center_x, center_y - triangle_size],
        [center_x - triangle_size/2, center_y + triangle_size/2],
        [center_x + triangle_size/2, center_y + triangle_size/2]
    ], color='red')
    axes[2].add_patch(triangle)
    

    for i in range(len(visibility)):
        if visibility[i] > 0.5:
            rel_pos = relative_positions[i].numpy()
            pixel_x = center_x + rel_pos[0] * img_size/2
            pixel_y = center_y - rel_pos[1] * img_size/2
            
            color = object_colors[i].numpy()
            circle = plt.Circle((pixel_x, pixel_y), 5, color=color)
            axes[2].add_patch(circle)
            

            axes[2].text(pixel_x + 10, pixel_y, f'({rel_pos[0]:.2f}, {rel_pos[1]:.2f})',
                        color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    axes[2].set_title('Rotated BEV\n(Agent facing up)')
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'frame_{sample["frame_id"]}_viz.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


    print(f"\nFrame {sample['frame_id']}:")
    print(f"Agent position: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})")
    print(f"Rotation angle: {np.degrees(rotation_angle):.1f}°")
    for i in range(len(visibility)):
        if visibility[i] > 0.5:
            abs_pos = absolute_positions[i].numpy()
            rel_pos = relative_positions[i].numpy()
            print(f"Target {i}:")
            print(f"  Absolute position: ({abs_pos[0]:.2f}, {abs_pos[1]:.2f})")
            print(f"  Relative position: ({rel_pos[0]:.2f}, {rel_pos[1]:.2f})")

def debug_dataset_rotations(maze_dirs, num_frames=20, save_dir=None):

    default_dir = Path(__file__).resolve().parents[1] / "debug" / "rotation"
    save_dir = Path(save_dir or os.environ.get("BEV_ROTATION_DEBUG_DIR", default_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    for maze_dir in maze_dirs:
        maze_name = os.path.basename(maze_dir)
        maze_save_dir = save_dir / maze_name
        maze_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing maze: {maze_name}")
        dataset = SingleMazeDataset(maze_dir)
        print(f"Dataset length: {len(dataset)}")

        for i in range(min(num_frames, len(dataset))):
            try:
                visualize_rotation_with_targets(dataset, i, maze_save_dir)
            except Exception as e:
                print(f"Error processing frame {i} in maze {maze_name}: {str(e)}")
                raise


def custom_collate_fn(batch):
    elem = batch[0]
    batch_dict = {}

    for key in elem:
        if key in ['frame_id', 'maze_id']:
            continue
        elif key == 'rotation_angle':
            batch_dict[key] = torch.tensor([d[key] for d in batch],
                                           dtype=torch.float32,
                                           pin_memory=True)
        else:
            batch_dict[key] = torch.stack([d[key] for d in batch],
                                          dim=0).contiguous()

    return batch_dict

def get_dataloaders(config, is_distributed=False, rank=0, world_size=1):

    maze_dirs = []
    for dataset_dir in config.DATASET_DIRS:
        dataset_path = os.path.join(config.DATA_ROOT, dataset_dir)
        maze_dirs.extend([
            os.path.join(dataset_path, d)
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])

    maze_dirs = sorted(maze_dirs)


    np.random.seed(42)
    np.random.shuffle(maze_dirs)


    n_total = len(maze_dirs)
    n_train = int(n_total * config.TRAIN_VAL_TEST_SPLIT[0])
    n_val = int(n_total * config.TRAIN_VAL_TEST_SPLIT[1])

    train_dirs = maze_dirs[:n_train]
    val_dirs = maze_dirs[n_train:n_train + n_val]
    test_dirs = maze_dirs[n_train + n_val:]


    def create_combined_dataset(maze_ids):
        datasets = []
        for maze_id in maze_ids:
            datasets.append(SingleMazeDataset(maze_id))
        return ConcatDataset(datasets)

    split_dirs = {
        'train': train_dirs,
        'val': val_dirs,
        'test': test_dirs
    }

    datasets = {
        split: create_combined_dataset(dirs)
        for split, dirs in split_dirs.items()
    }


    samplers = {}
    if is_distributed:
        samplers['train'] = DistributedSampler(
            datasets['train'],
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        samplers['val'] = DistributedSampler(
            datasets['val'],
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        samplers['test'] = DistributedSampler(
            datasets['test'],
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )


    dataloaders = {}
    for split in ['train', 'val', 'test']:
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=config.BATCH_SIZE,
            shuffle=(split == 'train' and not is_distributed),
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True,
            prefetch_factor=2,
            sampler=samplers.get(split) if is_distributed else None,
            collate_fn=custom_collate_fn,
            drop_last=split == 'train'
        )

    return dataloaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Debug rotation handling for BEV datasets.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.environ.get("BEV_DEBUG_BASE_DIR"),
        help="Base directory containing numbered maze folders (or set BEV_DEBUG_BASE_DIR).",
    )
    parser.add_argument(
        "--maze_ids",
        type=str,
        nargs="*",
        help="Specific maze folder names to visualise (default: all directories inside base_dir).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=50,
        help="Number of frames to export per maze.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Optional output directory for debug plots (defaults to ./debug/rotation).",
    )
    args = parser.parse_args()

    if not args.base_dir:
        parser.error("Please provide --base_dir or set the BEV_DEBUG_BASE_DIR environment variable.")

    base_path = Path(args.base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    if args.maze_ids:
        maze_dirs = [str(base_path / maze_id) for maze_id in args.maze_ids]
    else:
        maze_dirs = [str(p) for p in base_path.iterdir() if p.is_dir()]

    print("Starting debug visualization...")
    debug_dataset_rotations(maze_dirs, num_frames=args.num_frames, save_dir=args.save_dir)
    print("Finished debug visualization!")
