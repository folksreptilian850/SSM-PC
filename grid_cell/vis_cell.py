import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from models.toroidal_grid_cell import GridCellNetwork
from models.place_hd_cells import PlaceCellEnsemble, HeadDirectionCellEnsemble
from datasets.navigation_dataset import SingleMazeDataset

def build_and_load_model(checkpoint_path, config):



    place_cells = PlaceCellEnsemble(
        n_cells=config.PLACE_CELLS_N,
        scale=config.PLACE_CELLS_SCALE,
        pos_min=0,
        pos_max=config.ENV_SIZE,
        seed=config.SEED
    )
    hd_cells = HeadDirectionCellEnsemble(
        n_cells=config.HD_CELLS_N,
        concentration=config.HD_CELLS_CONCENTRATION,
        seed=config.SEED
    )
    model = GridCellNetwork(
        place_cells=place_cells,
        hd_cells=hd_cells,
        input_size=3,
        hidden_size=config.HIDDEN_SIZE,
        bottleneck_size=256,
        dropout_rate=config.DROPOUT_RATE
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, place_cells, hd_cells

def load_single_trajectory(trajectory_dir, sequence_length=100, stride=1):



    dataset = SingleMazeDataset(trajectory_dir, sequence_length=sequence_length, stride=stride)
    if len(dataset) == 0:
        raise ValueError(f"轨迹 {trajectory_dir} 没有足够连续的帧数据！")
    sample = dataset[0]
    positions = sample['positions']            # [S, 2]
    angles = sample['angles']                  # [S]
    velocities = sample['velocities']          # [S, 2]
    angular_vels = sample['angular_velocities']# [S]
    return positions, angles, velocities, angular_vels

def prepare_inference_input(positions, angles, velocities, angular_vels):






    w = angular_vels.unsqueeze(-1)  # [S,1]
    velocity_input = torch.cat([velocities, w], dim=-1).unsqueeze(0)  # [1,S,3]
    init_pos = positions[0].unsqueeze(0)  # [1,2]
    init_hd = angles[0].unsqueeze(0)        # [1]
    return velocity_input, init_pos, init_hd

def decode_predictions(place_logits, hd_logits, place_cells, hd_cells):











            
    place_probs = torch.softmax(place_logits, dim=-1)  # [S, n_place]
    hd_probs = torch.softmax(hd_logits, dim=-1)  # [S, n_hd]

                                                     
                                                           
    cell_centers = place_cells.means.to(place_probs.device)  # [n_place,2]
    pred_positions = torch.matmul(place_probs, cell_centers)  # [S,2]
    
                              
    hd_means = hd_cells.means.to(hd_probs.device)  # [n_hd]
                  
    hd_complex = torch.exp(1j * hd_means)  # [n_hd]
              
    pred_hd_complex = torch.sum(hd_probs * hd_complex, dim=-1)  # [S]
    pred_angles = torch.angle(pred_hd_complex)  # [S]
    return pred_positions, pred_angles

def visualize_trajectory(gt_positions, pred_positions, gt_angles, pred_angles,
                         save_path, maze_size=15):






                                    
    if len(gt_positions) > 1:
        pos_errors = np.linalg.norm(gt_positions[1:] - pred_positions[1:], axis=1)
        mean_pos_error = np.mean(pos_errors)
                         
        angle_errors = np.abs(np.angle(np.exp(1j * (gt_angles[1:] - pred_angles[1:]))))
        mean_angle_error = np.mean(angle_errors)
    else:
        mean_pos_error = 0.0
        mean_angle_error = 0.0

    plt.figure(figsize=(10, 8))

                            
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], 'o-', color='blue',
             label="GT Trajectory", markersize=3, alpha=0.7)

                                  
    plt.plot(gt_positions[0, 0], gt_positions[0, 1], 'bo', markersize=10, label='GT/Pred Start (Given)')
    plt.plot(gt_positions[-1, 0], gt_positions[-1, 1], 'b^', markersize=10, label='GT End')

                           
    plt.plot(pred_positions[:, 0], pred_positions[:, 1], 'x-', color='orange',
             label="Predicted Trajectory", markersize=3)

                      
                                
    plt.plot(pred_positions[-1, 0], pred_positions[-1, 1], '^', color='orange',
             markersize=10, label='Pred End')

             
    plt.xlim(0, maze_size)
    plt.ylim(0, maze_size)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Trajectory Comparison (Error Excludes t=0)\n"
              f"Mean Pos Error: {mean_pos_error:.3f}, Mean Rot Error: {mean_angle_error:.3f}")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trajectory visualization saved to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predicted and ground-truth trajectories for a single maze run.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained grid-cell checkpoint.")
    parser.add_argument("--trajectory", type=Path, required=True, help="Path to the trajectory folder (contains frame_info.json).")
    parser.add_argument("--output", type=Path, default=Path("trajectory_comparison.png"), help="Output image path for the trajectory comparison plot.")
    parser.add_argument("--sequence_length", type=int, default=100, help="Sequence length to evaluate.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride when sampling the trajectory.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.trajectory.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {args.trajectory}")

    model, place_cells, hd_cells = build_and_load_model(str(args.checkpoint), config)

    positions, angles, velocities, angular_vels = load_single_trajectory(
        str(args.trajectory), sequence_length=args.sequence_length, stride=args.stride
    )
    print(f"Loaded trajectory with {positions.shape[0]} frames.")

    velocity_input, init_pos, init_hd = prepare_inference_input(positions, angles, velocities, angular_vels)

                          
    gt_positions = positions.cpu().numpy()  # [S,2]
    gt_angles = angles.cpu().numpy()  # [S]
    
                      
    with torch.no_grad():
        outputs = model(velocity_input, init_pos, init_hd)
        place_logits = outputs['place_logits'].squeeze(0)  # [S, n_place]
        hd_logits = outputs['hd_logits'].squeeze(0)  # [S, n_hd]
        
                    
        pred_positions, pred_angles = decode_predictions(place_logits, hd_logits, place_cells, hd_cells)
        pred_positions = pred_positions.cpu().numpy()  # [S,2]
        pred_angles = pred_angles.cpu().numpy()  # [S]
    
                   
    pred_positions[0] = gt_positions[0]              
    pred_angles[0] = gt_angles[0]              
    
                              
    print(f"GT positions shape: {gt_positions.shape}, Pred positions shape: {pred_positions.shape}")
    visualize_trajectory(gt_positions, pred_positions, gt_angles, pred_angles, str(args.output.resolve()))
    
                             
    if len(gt_positions) > 1:
        pos_errors = np.linalg.norm(gt_positions[1:] - pred_positions[1:], axis=1)
        mean_pos_error = np.mean(pos_errors)
        max_pos_error = np.max(pos_errors)
              
        angle_errors = np.abs(np.angle(np.exp(1j * (gt_angles[1:] - pred_angles[1:]))))
        mean_angle_error = np.mean(angle_errors)
        max_angle_error = np.max(angle_errors)
        
        print(f"评估指标 (排除t=0):")
        print(f"  位置误差: 平均 {mean_pos_error:.3f}m, 最大 {max_pos_error:.3f}m")
        print(f"  角度误差: 平均 {mean_angle_error:.3f}rad, 最大 {max_angle_error:.3f}rad")
        print(f"  角度误差: 平均 {np.degrees(mean_angle_error):.1f}°, 最大 {np.degrees(max_angle_error):.1f}°")


if __name__ == "__main__":
    main()
