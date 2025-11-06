import argparse
import json
import os
import shutil
import sys
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from datasets.maze_dataset import SingleMazeDataset
from models.bev_net import BEVPredictor
from models.target_detector import TargetDetector

warnings.filterwarnings("ignore")
os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'

TARGET_COLORS = [
    (170, 38, 30),   # red
    (99, 170, 88),   # green
    (39, 140, 217),  # blue
    (93, 105, 199),  # purple
    (220, 193, 59),  # yellow
    (220, 128, 107), # salmon
]

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory)
        print(f"Cleared directory: {directory}")
    else:
        print(f"Directory does not exist: {directory}")
        os.makedirs(directory)

def process_bev_prediction(pred_bev):

    rgb = pred_bev[:3]
    alpha = pred_bev[3]
    
    road_mask = alpha > 0.6
    rgb_output = np.zeros_like(rgb)
    gray_value = 102
    
    for c in range(3):
        rgb_output[c][road_mask] = gray_value
        rgb_output[c][~road_mask] = 0
    
    output = np.concatenate([rgb_output, alpha[None, ...] * 255], axis=0)
    output = output.transpose(1, 2, 0).astype(np.uint8)
    
    return output

def write_video(frames, filename, size):

    if len(frames) == 0:
        print(f"Warning: No frames to write for {filename}.")
        return
    
    print(f"Writing {len(frames)} frames to {filename}")
    writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        10.0,
        size
    )
    
    for i, frame in enumerate(frames):
        try:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            writer.write(frame)
        except Exception as e:
            print(f"Error writing frame {i}: {e}")
            
    writer.release()
    print(f"Successfully wrote video to {filename}")

def combine_videos_cv2(output_dir):

    input_files = ['fps_views.mp4', 'pred_bev.mp4', 'gt_bev.mp4']
    video_paths = [os.path.join(output_dir, f) for f in input_files]
    
              
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Error: Missing video file {path}")
            return
    
            
    videos = [cv2.VideoCapture(path) for path in video_paths]
    
           
    success_flags = []
    frames = []
    for cap in videos:
        success, frame = cap.read()
        success_flags.append(success)
        frames.append(frame)
    
                       
    if not all(success_flags):
        print("Error: Cannot read first frame from all videos")
        return
    
    target_size = (250, 250)
    frames = [cv2.resize(f, target_size, interpolation=cv2.INTER_NEAREST) for f in frames]
    
            
    writer = cv2.VideoWriter(
        os.path.join(output_dir, 'combined_output.mp4'),
        cv2.VideoWriter_fourcc(*'XVID'),
        10.0,
        (750, 250)                  
    )
    
           
    while True:
                    
        if any(f is None for f in frames):
            break
            
                
        combined_frame = np.hstack(frames)
        writer.write(combined_frame)
        
               
        success_flags = []
        next_frames = []
        for cap in videos:
            success, frame = cap.read()
            success_flags.append(success)
            next_frames.append(frame if success else None)
            
                           
        if not all(success_flags):
            break
            
               
        frames = [cv2.resize(f, target_size, interpolation=cv2.INTER_NEAREST) for f in next_frames]
    
          
    for cap in videos:
        cap.release()
    writer.release()
    
    print(f"Combined video saved to {os.path.join(output_dir, 'combined_output.mp4')}")

def visualize_inference(bev_model_path, target_model_path, dataset_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

            
    fps_frames = []
    pred_frames = []
    gt_frames = []
    
               
    config = Config()
    bev_model = BEVPredictor(config).to(device)
    bev_model.load_state_dict(torch.load(bev_model_path, map_location=device)['model_state_dict'])
    bev_model.eval()

              
    target_model = TargetDetector(max_objects=config.MAX_OBJECTS).to(device)
    target_model.load_state_dict(torch.load(target_model_path, map_location=device)['model_state_dict'])
    target_model.eval()

    dataset = SingleMazeDataset(dataset_path)
    print(f"Dataset size: {len(dataset)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, num_workers=4, pin_memory=True
        )

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            fps_view = batch['fps_view'].to(device)
            bev_view = batch['bev_view']
            angle = batch['angle'].to(device)
            object_colors = batch['object_colors'].to(device)

                     
            bev_outputs = bev_model(fps_view, angle)
            
                      
            target_outputs = target_model(fps_view, angle)
            
            batch_size = fps_view.shape[0]
            for i in range(batch_size):
                          
                fps_img = batch['fps_view'][i].cpu().permute(1, 2, 0).numpy()
                fps_img = (fps_img * np.array([0.229, 0.224, 0.225]) + 
                          np.array([0.485, 0.456, 0.406])) * 255
                fps_img = fps_img.astype(np.uint8)
                fps_frames.append(cv2.cvtColor(fps_img, cv2.COLOR_RGB2BGR))

                         
                true_bev = batch['bev_view'][i].cpu().permute(1, 2, 0).numpy() * 255
                true_bev = true_bev.astype(np.uint8)
                gt_frames.append(cv2.cvtColor(true_bev, cv2.COLOR_RGBA2BGR))

                                   
                pred_bev = bev_outputs['bev'][i].cpu().numpy()
                pred_bev = process_bev_prediction(pred_bev)
                pred_bev_with_objects = pred_bev.copy()

                          
                h, w = pred_bev.shape[:2]
                center = np.array([w/2, h/2])

                                 
                pred_visibility = target_outputs['object_visibility'][i].cpu().numpy()
                pred_positions = target_outputs['object_relative_positions'][i].cpu().numpy()

                         
                for obj_idx in range(len(pred_visibility)):
                    if pred_visibility[obj_idx] > 0.7:
                        rel_pos = pred_positions[obj_idx]
                        rel_pos[1] = -rel_pos[1]        
                        pixel_pos = center + (rel_pos * (min(h, w)/2))
                        pixel_pos = np.clip(pixel_pos, [0, 0], [w-1, h-1])
                        
                        color = TARGET_COLORS[obj_idx % len(TARGET_COLORS)]
                        cv2.circle(pred_bev_with_objects, 
                                tuple(pixel_pos.astype(int)), 
                                5, color, -1)
                        cv2.circle(pred_bev_with_objects, 
                                tuple(pixel_pos.astype(int)), 
                                7, color, 2)

                pred_frames.append(cv2.cvtColor(pred_bev_with_objects, cv2.COLOR_RGBA2BGR))

                            
                if batch_idx == 0 and i == 0:
                    debug_dir = os.path.join(output_dir, 'debug')
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    cv2.imwrite(os.path.join(debug_dir, 'first_fps.png'), 
                              cv2.cvtColor(fps_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(debug_dir, 'first_gt.png'), 
                              cv2.cvtColor(true_bev, cv2.COLOR_RGBA2BGR))
                    cv2.imwrite(os.path.join(debug_dir, 'first_pred.png'),
                              cv2.cvtColor(pred_bev_with_objects, cv2.COLOR_RGBA2BGR))
                    
                            
                    detection_results = {
                        'pred_visibility': pred_visibility.tolist(),
                        'pred_positions': pred_positions.tolist(),
                        'true_visibility': batch['object_visibility'][i].numpy().tolist(),
                        'true_positions': batch['object_relative_positions'][i].numpy().tolist()
                    }
                    with open(os.path.join(debug_dir, 'first_frame_detections.json'), 'w') as f:
                        json.dump(detection_results, f, indent=2)

    print(f"Collected frames: FPS={len(fps_frames)}, Pred={len(pred_frames)}, GT={len(gt_frames)}")

          
    print("Generating videos...")
    write_video(fps_frames, os.path.join(output_dir, 'fps_views.mp4'), (250, 250))
    write_video(pred_frames, os.path.join(output_dir, 'pred_bev.mp4'), (250, 250))
    write_video(gt_frames, os.path.join(output_dir, 'gt_bev.mp4'), (250, 250))

          
    print("Combining videos...")
    combine_videos_cv2(output_dir)

    print("Visualization completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BEV predictor inference and export visualization videos.")
    parser.add_argument(
        "--bev_model",
        type=str,
        default=os.environ.get("BEV_MODEL_PATH"),
        help="Path to the trained BEV predictor checkpoint (set BEV_MODEL_PATH env var to avoid passing explicitly).",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default=os.environ.get("TARGET_MODEL_PATH"),
        help="Path to the trained target detector checkpoint (or set TARGET_MODEL_PATH env var).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.environ.get("BEV_DATASET_PATH"),
        help="Path to a dataset sequence folder containing fps/bev frames (or set BEV_DATASET_PATH env var).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("BEV_OUTPUT_DIR", "visualization_output"),
        help="Directory where rendered videos will be stored.",
    )
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="Remove the output directory before writing new visualizations.",
    )
    args = parser.parse_args()

    missing = [
        flag
        for flag, value in {
            "--bev_model": args.bev_model,
            "--target_model": args.target_model,
            "--dataset": args.dataset,
        }.items()
        if not value
    ]
    if missing:
        parser.error(f"The following required arguments are missing (or unset env vars): {', '.join(missing)}")

    if args.clean_output:
        clear_directory(args.output_dir)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    visualize_inference(args.bev_model, args.target_model, args.dataset, args.output_dir)
