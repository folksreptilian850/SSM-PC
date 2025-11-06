# bev_mapping.py
import os
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


def process_bev_prediction(pred_bev):
    """
    Process BEV prediction result with improved path detection.
    Modified to better match the processing in paste-2.txt.
    """
    rgb = pred_bev[:3]
    alpha = pred_bev[3]
    
    # More lenient road mask threshold to catch more potential paths
    road_mask = alpha > 0.3  # Lower threshold from 0.6 to 0.3
    rgb_output = np.zeros_like(rgb)
    gray_value = 102  # Keep the same gray value for consistency
    
    for c in range(3):
        rgb_output[c][road_mask] = gray_value
        rgb_output[c][~road_mask] = 0
    
    # Create RGBA output
    output = np.concatenate([rgb_output, alpha[None, ...] * 255], axis=0)
    output = output.transpose(1, 2, 0).astype(np.uint8)
    
    return output


def load_frame_info(dataset_path):
    """
    Load the frame_info.json file and convert to the expected format
    """
    frame_info_path = os.path.join(dataset_path, 'frame_info.json')
    with open(frame_info_path, 'r') as f:
        frame_info_raw = json.load(f)

    # Convert to the expected format
    frame_info = {}
    for frame_id, data in frame_info_raw.items():
        # Extract position and rotation from agent_state
        agent_position = data['agent_state']['position']
        rotation_angle = data['agent_state']['rotation']

        # Create formatted entry
        frame_info[frame_id] = {
            'agent_position': agent_position,
            'rotation_angle': rotation_angle,
            'visible_indices': data['observations'].get('visible_targets', []),
            'frame_id': data['frame_id']
        }

    return frame_info


def load_frames(dataset_path, max_frames=None):
    """
    Load frames directly without using SingleMazeDataset.
    Optimized for performance with removed debugging.
    """
    import os
    import numpy as np
    import torch
    from PIL import Image
    import json
    import time
    
    start_time = time.time()
    
    # Load frame_info with memoization
    if not hasattr(load_frames, 'frame_info_cache'):
        load_frames.frame_info_cache = {}
    
    if dataset_path not in load_frames.frame_info_cache:
        frame_info_path = os.path.join(dataset_path, 'frame_info.json')
        with open(frame_info_path, 'r') as f:
            frame_info_raw = json.load(f)
            
        # Convert to the expected format
        frame_info = {}
        for frame_id, data in frame_info_raw.items():
            agent_position = data['agent_state']['position']
            rotation_angle = data['agent_state']['rotation']
            
            # Create formatted entry
            frame_info[frame_id] = {
                'agent_position': agent_position,
                'rotation_angle': rotation_angle,
                'visible_indices': data['observations'].get('visible_targets', []),
                'frame_id': data['frame_id']
            }
            
        load_frames.frame_info_cache[dataset_path] = frame_info
    else:
        frame_info = load_frames.frame_info_cache[dataset_path]

    # Get sorted frame IDs
    frame_ids = sorted(frame_info.keys(), key=lambda x: int(x.split('_')[1]))
    if max_frames:
        frame_ids = frame_ids[:max_frames]

    # Prepare directories
    fps_dir = os.path.join(dataset_path, 'fps_views')
    bev_dir = os.path.join(dataset_path, 'bev_views')

    # Load frames
    frames = []
    for frame_id in frame_ids:
        # Frame info
        info = frame_info[frame_id]

        # Load FPS image
        fps_path = os.path.join(fps_dir, f"{frame_id}.png")
        if not os.path.exists(fps_path):
            fps_path = os.path.join(fps_dir, f"frame_{frame_id}.png")

        if not os.path.exists(fps_path):
            print(f"Warning: FPS image not found for frame {frame_id}")
            continue

        # Load BEV image
        bev_path = os.path.join(bev_dir, f"{frame_id}.png")
        if not os.path.exists(bev_path):
            bev_path = os.path.join(bev_dir, f"frame_{frame_id}.png")

        if not os.path.exists(bev_path):
            print(f"Warning: BEV image not found for frame {frame_id}")
            continue
            
        # Load images
        fps_img = Image.open(fps_path).convert('RGB')
        fps_tensor = torch.from_numpy(np.array(fps_img)).float() / 255.0
        fps_tensor = fps_tensor.permute(2, 0, 1)
        
        bev_img = Image.open(bev_path).convert('RGBA')
        bev_tensor = torch.from_numpy(np.array(bev_img)).float() / 255.0
        bev_tensor = bev_tensor.permute(2, 0, 1)

        # Create frame data
        frame_data = {
            'frame_id': frame_id,
            'fps_view': fps_tensor,
            'bev_view': bev_tensor,
            'agent_position': info['agent_position'],
            'rotation_angle': info['rotation_angle']
        }

        frames.append(frame_data)

    elapsed = time.time() - start_time
    print(f"Loaded {len(frames)} frames from {dataset_path} in {elapsed:.2f} seconds")
    return frames


def visualize_agent_bev_mapping(dataset_path, output_dir, model_path=None, max_frames=200, frame_interval=5):
    """
    Visualizes how BEV predictions map onto the 15x15 grid.
    Modified to better handle model predictions based on paste-2.txt visualization approach.
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_dir (str): Directory to save output files
        model_path (str, optional): Path to the BEV model
        max_frames (int, optional): Maximum number of frames to process
        frame_interval (int, optional): Interval between frames to save mapped images
    """
    import os
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    import time
    
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load frames
    frames = load_frames(dataset_path, max_frames=max_frames)

    # If model path is provided, load the model
    if model_path:
        try:
            from models.bev_net import BEVPredictor
            from config import Config
            config = Config()
            model = BEVPredictor(config).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
            model.eval()
            print("Successfully loaded BEV model from:", model_path)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            model_path = None

    # Pre-allocate grid image template to reuse
    grid_template = np.zeros((250, 250, 3), dtype=np.uint8)
    grid_template[:, :] = (50, 50, 50)  # Set a dark gray background

    # Pre-draw grid lines once
    for i in range(16):
        pos = int(i * (250 / 15))
        cv2.line(grid_template, (0, pos), (250, pos), (100, 100, 100), 1)
        cv2.line(grid_template, (pos, 0), (pos, 250), (100, 100, 100), 1)
    
    # Track which frames are actually saved
    saved_frame_indices = []
    
    # Process frames
    for frame_idx, frame in enumerate(tqdm(frames, desc="Processing frames")):
        # Only process if this frame should be saved (every frame_interval frames)
        if frame_idx % frame_interval == 0 or frame_idx == len(frames) - 1:
            # Add this frame index to the saved list for SLAM reconstruction
            saved_frame_indices.append(frame_idx)
            
            # Copy the template grid for this frame
            mapped_img = grid_template.copy()
            
            # Get agent position and rotation
            agent_position = frame['agent_position']
            agent_rotation = frame['rotation_angle']
    
            # Prepare inputs for model if needed
            fps_view = frame['fps_view'].unsqueeze(0).to(device)
    
            # Use ground truth BEV or model prediction
            if model_path:
                # Prepare angle tensor for the model - format from paste-2.txt
                angle_rad = agent_rotation
                angle_tensor = torch.tensor([
                    [np.cos(angle_rad), np.sin(angle_rad)]
                ], dtype=torch.float32).to(device)
    
                with torch.no_grad():
                    outputs = model(fps_view, angle_tensor)
                    pred_bev = outputs['bev'][0].cpu().numpy()
                    pred_bev = process_bev_prediction(pred_bev)
                    
                    # Debug output for the first frame
                    if frame_idx == 0:
                        debug_dir = os.path.join(output_dir, 'debug')
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        # Save raw model output
                        raw_pred = outputs['bev'][0].cpu().numpy()
                        np.save(os.path.join(debug_dir, 'raw_model_output.npy'), raw_pred)
                        
                        # Save processed BEV (before mapping)
                        cv2.imwrite(os.path.join(debug_dir, 'processed_bev.png'), 
                                   cv2.cvtColor(pred_bev, cv2.COLOR_RGBA2BGR))
            else:
                # Use ground truth BEV
                true_bev = frame['bev_view'].cpu().permute(1, 2, 0).numpy() * 255
                pred_bev = true_bev.astype(np.uint8)
    
            # Calculate agent's grid cell position
            grid_cell_x = int(agent_position[0])
            grid_cell_y = int(agent_position[1])
            cell_center_x = grid_cell_x + 0.5
            cell_center_y = grid_cell_y + 0.5
            agent_pixel_x = int(cell_center_x / 15 * 250)
            agent_pixel_y = int(cell_center_y / 15 * 250)
    
            # Draw agent as a circle with a line indicating orientation
            cv2.circle(mapped_img, (agent_pixel_x, agent_pixel_y), 5, (0, 0, 255), -1)
    
            # Draw orientation arrow (matching paste-2.txt approach)
            # arrow_length = 20
            # # Convert rotation to match world coordinates
            # end_x = agent_pixel_x + arrow_length * np.cos(agent_rotation)
            # end_y = agent_pixel_y + arrow_length * np.sin(agent_rotation)
            # cv2.arrowedLine(mapped_img, (agent_pixel_x, agent_pixel_y),
            #               (int(end_x), int(end_y)), (0, 255, 0), 2)
    
            # Extract alpha channel and prepare for blending
            if pred_bev.shape[2] == 4:  # RGBA image
                alpha = pred_bev[:, :, 3]
                bev_bgr = cv2.cvtColor(pred_bev, cv2.COLOR_RGBA2BGR)
            else:  # RGB image
                bev_bgr = pred_bev.copy()
                alpha = np.ones((pred_bev.shape[0], pred_bev.shape[1]), dtype=np.uint8) * 255
    
            # Calculate the placement coordinates exactly like in paste-2.txt
            grid_cell_height = 250 / 15
            top_left_x = int(agent_pixel_x - 125 - 0.5 * grid_cell_height)
            top_left_y = int(agent_pixel_y - 125 - 0.5 * grid_cell_height)
            
            # Define regions and perform bounds checking
            roi_x1 = max(0, top_left_x)
            roi_y1 = max(0, top_left_y)
            roi_x2 = min(250, top_left_x + 250)
            roi_y2 = min(250, top_left_y + 250)
            
            bev_x1 = max(0, -top_left_x)
            bev_y1 = max(0, -top_left_y)
            bev_x2 = bev_x1 + (roi_x2 - roi_x1)
            bev_y2 = bev_y1 + (roi_y2 - roi_y1)
    
            # Apply alpha blending with improved path detection
            if roi_x2 > roi_x1 and roi_y2 > roi_y1 and bev_x2 > bev_x1 and bev_y2 > bev_y1:
                # Extract regions
                bev_roi = bev_bgr[bev_y1:bev_y2, bev_x1:bev_x2]
                alpha_roi = alpha[bev_y1:bev_y2, bev_x1:bev_x2]
                
                # Create alpha mask with lower threshold to catch more potential paths
                alpha_mask = alpha_roi > 30  # More lenient threshold
                
                # Apply blending with specific path color
                for c in range(3):
                    # Get the target region
                    channel = mapped_img[roi_y1:roi_y2, roi_x1:roi_x2, c]
                    # Set path color (gray value 120) for all detected path areas
                    channel[alpha_mask] = 120
                    # Update the channel in the grid image
                    mapped_img[roi_y1:roi_y2, roi_x1:roi_x2, c] = channel
                
                # Debug output for the first frame
                if frame_idx == 0:
                    debug_dir = os.path.join(output_dir, 'debug')
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Save ROI and mask
                    cv2.imwrite(os.path.join(debug_dir, 'bev_roi.png'), bev_roi)
                    cv2.imwrite(os.path.join(debug_dir, 'alpha_mask.png'), 
                               alpha_mask.astype(np.uint8) * 255)
            
            # Save the mapped image for SLAM reconstruction
            cv2.imwrite(os.path.join(output_dir, f'frame_{frame_idx}_mapped.png'), mapped_img)
    
    # Save the list of processed frame indices for the SLAM reconstruction to use
    import json
    with open(os.path.join(output_dir, 'processed_frames.json'), 'w') as f:
        json.dump(saved_frame_indices, f)
    
    elapsed = time.time() - start_time
    print(f"BEV mapping visualization completed in {elapsed:.2f} seconds")
    print(f"Generated {len(saved_frame_indices)} mapped frames at interval {frame_interval}")
    print(f"BEV mapping saved to {output_dir}")

def visualize_slam_reconstruction(dataset_path, output_dir, model_path=None, max_frames=100):
    """
    Implements a SLAM-like reconstruction by accumulating frames from frame_xx_mapped.png.
    This function uses the sparse output of visualize_agent_bev_mapping and combines 
    the frames to generate a global map.
    """
    import os
    import cv2
    import numpy as np
    from tqdm import tqdm
    import json
    import time
    
    start_time = time.time()
    print(f"\nStarting SLAM reconstruction...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if processed_frames.json exists to know which frames were saved
    processed_frames_path = os.path.join(output_dir, 'processed_frames.json')
    if os.path.exists(processed_frames_path):
        with open(processed_frames_path, 'r') as f:
            frame_indices = json.load(f)
        print(f"Found {len(frame_indices)} sparse frame indices from previous BEV mapping")
    else:
        # If no processed_frames.json, check for any frame_xx_mapped.png files
        frame_files = [f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('_mapped.png')]
        
        if not frame_files:
            print("No frame_xx_mapped.png files found. Running visualize_agent_bev_mapping first...")
            bev_mapping_dir = output_dir  # Use the same directory
            # Default to every 5th frame if we're generating them now
            visualize_agent_bev_mapping(dataset_path, bev_mapping_dir, model_path, max_frames=max_frames, frame_interval=5)
            
            # Re-check for processed_frames.json
            if os.path.exists(processed_frames_path):
                with open(processed_frames_path, 'r') as f:
                    frame_indices = json.load(f)
            else:
                # If still no processed_frames.json, re-scan for files
                frame_files = [f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('_mapped.png')]
                frame_indices = [int(f.split('_')[1].split('.')[0]) for f in frame_files]
                frame_indices.sort()
        else:
            # Extract frame indices from filenames
            frame_indices = [int(f.split('_')[1].split('.')[0]) for f in frame_files]
            frame_indices.sort()
    
    if not frame_indices:
        print("Error: No frame data found for SLAM reconstruction.")
        return
    
    print(f"Using {len(frame_indices)} frame indices for SLAM reconstruction.")
    
    # Load the first frame to get dimensions and initialize the global map
    first_frame_path = os.path.join(output_dir, f'frame_{frame_indices[0]}_mapped.png')
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"Error: Could not read first frame from {first_frame_path}")
        return
    
    h, w = first_frame.shape[:2]
    
    # Create global map - make it 2x the size of individual frames to ensure enough space
    global_map_size = max(h, w) * 2
    global_map = np.ones((global_map_size, global_map_size, 3), dtype=np.uint8) * 50  # Dark gray background
    
    # Draw grid lines in the global map
    grid_cell_size = h / 15  # Assuming individual frames have a 15x15 grid
    for i in range(int(global_map_size / grid_cell_size) + 1):
        pos = int(i * grid_cell_size)
        cv2.line(global_map, (0, pos), (global_map_size, pos), (100, 100, 100), 1)
        cv2.line(global_map, (pos, 0), (pos, global_map_size), (100, 100, 100), 1)
    
    # Center position for the first frame
    center_x = global_map_size // 2
    center_y = global_map_size // 2
    
    # Load frames data for trajectory information
    frames_data = load_frames(dataset_path, max_frames=max(frame_indices) + 1)
    
    # Create trajectory from all frames, not just the sparse ones
    trajectory = []
    for idx, frame_data in enumerate(frames_data):
        if idx <= max(frame_indices):  # Only include frames up to our max processed frame
            trajectory.append((frame_data['agent_position'][0], frame_data['agent_position'][1]))
    
    # Create a confidence map for building the SLAM map
    confidence_map = np.zeros((global_map_size, global_map_size), dtype=np.float32)
    
    # Process each mapped frame and add to global map
    # Use frame_indices instead of all frames
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Building SLAM map")):
        # Load the mapped frame
        frame_path = os.path.join(output_dir, f'frame_{frame_idx}_mapped.png')
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        # Convert to grayscale for path detection (more efficient)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find path areas (gray pixels with value around 120)
        path_mask = (frame_gray > 70) & (frame_gray < 170)
        
        # Scale to global map size
        scale_factor = global_map_size / h
        if scale_factor != 1.0:
            # Only resize the mask (more efficient than resizing the full image)
            path_mask_resized = cv2.resize(path_mask.astype(np.uint8), 
                                           (global_map_size, global_map_size), 
                                           interpolation=cv2.INTER_NEAREST)
            path_mask = path_mask_resized > 0
        
        # Update confidence for path areas
        confidence_map[path_mask] += 0.7
        
        # Apply a small confidence decay to all cells
        if i % 5 == 0:  # Only decay occasionally
            confidence_map *= 0.95
        
        # Update global map where confidence exceeds threshold
        path_areas = confidence_map > 0.3
        global_map[path_areas] = (120, 120, 120)  # Path color
        
        # Create visualization every 10 indices or for the last frame
        if i % 10 == 0 or i == len(frame_indices) - 1:
            # Create a copy for visualization
            vis_map = global_map.copy()
            
            # Draw trajectory
            draw_trajectory_quantized(vis_map, trajectory, maze_size=15, global_size=global_map_size)
            
            # Save the SLAM map
            cv2.imwrite(os.path.join(output_dir, f'slam_map_frame_{i:03d}.png'), vis_map)
    
    # Create a final SLAM map with the full trajectory
    final_map = global_map.copy()
    draw_trajectory_quantized(final_map, trajectory, maze_size=15, global_size=global_map_size)
    cv2.imwrite(os.path.join(output_dir, 'slam_map_final.png'), final_map)
    
    # Create video of the SLAM reconstruction process
    create_slam_video(output_dir, len(frame_indices), fps=10)
    
    elapsed = time.time() - start_time
    print(f"SLAM reconstruction completed in {elapsed:.2f} seconds")
    print(f"SLAM reconstruction saved to {output_dir}")


def draw_trajectory_quantized(map_image, trajectory, maze_size=15, global_size=500):
    """
    Draws the agent trajectory on the map, with positions quantized to grid cells.
    Optimized for performance without debug outputs.
    """
    if not trajectory:
        return

    # Scale factor from maze coordinates to global map coordinates
    scale = global_size / maze_size

    # More efficient trajectory processing using sets
    pixel_trajectory = []
    cell_centers = set()

    for x, y in trajectory:
        # Round to nearest grid cell
        grid_x = round(x)
        grid_y = round(y)

        # Use cell center for drawing
        cell_key = f"{grid_x},{grid_y}"
        if cell_key not in cell_centers:
            cell_centers.add(cell_key)
            # Add 0.5 to get the center of the cell
            pixel_x = int((grid_x + 0.5) * scale)
            pixel_y = int((grid_y + 0.5) * scale)
            pixel_trajectory.append((pixel_x, pixel_y))

    # Draw the trajectory line
    for i in range(1, len(pixel_trajectory)):
        cv2.line(map_image, pixel_trajectory[i - 1], pixel_trajectory[i], (0, 0, 255), 2)

    # Draw start and end points
    if pixel_trajectory:
        cv2.circle(map_image, pixel_trajectory[0], 5, (0, 255, 0), -1)  # Green start
        cv2.circle(map_image, pixel_trajectory[-1], 5, (255, 0, 0), -1)  # Red end



# Optimized create_slam_video
def create_slam_video(output_dir, num_frames, fps=10):
    """
    Creates a video from the SLAM map frames.
    Optimized without debug outputs.
    """
    import os
    import cv2
    import time
    
    start_time = time.time()
    
    # Get frames from slam_map_frame_xxx.png files
    frames = []
    frame_indices = list(range(0, num_frames, 10)) + [num_frames - 1]
    if num_frames - 1 not in frame_indices:
        frame_indices.append(num_frames - 1)
    
    if not frame_indices:
        frame_indices = [0]

    for i in sorted(frame_indices):
        frame_path = os.path.join(output_dir, f'slam_map_frame_{i:03d}.png')
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)

    if not frames:
        print("No frames found for video creation.")
        return

    output_path = os.path.join(output_dir, 'slam_reconstruction.mp4')

    # Create video writer
    height, width = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # Write frames
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    elapsed = time.time() - start_time
    print(f"Video created in {elapsed:.2f} seconds and saved to {output_path}")

def transform_bev_to_global(bev_image, agent_position, agent_rotation, maze_size=15, global_size=500):
    """
    Transforms a BEV image to global map coordinates with corrected rotations.
    """
    # Convert BEV to BGRA if it's BGR
    if bev_image.shape[2] == 3:
        bev_bgra = cv2.cvtColor(bev_image, cv2.COLOR_BGR2BGRA)
        bev_bgra[:, :, 3] = 255  # Set alpha to fully opaque
    else:
        bev_bgra = bev_image.copy()

    # Scale factor from maze coordinates to global map coordinates
    scale = global_size / maze_size

    # CORRECTED: Calculate agent position in global map coordinates
    # Direct mapping without y-flip
    agent_global_x = agent_position[0] * scale
    agent_global_y = agent_position[1] * scale

    grid_cell_height = global_size / maze_size
    tx_float = agent_global_x - 125 - 0.5 * grid_cell_height
    ty_float = agent_global_y - 125 - 0.5 * grid_cell_height

    # Convert to integers for array indexing
    tx = int(tx_float)
    ty = int(ty_float)

    # Create a larger canvas for the transformation
    canvas = np.zeros((global_size, global_size, 4), dtype=np.uint8)

    # CORRECTED: Calculate the rotation to match the coordinate system
    # Apply rotation-π/2
    # Use the original BEV as is
    rotated_bev = bev_bgra.copy()

    # Calculate the bounds for placing the rotated BEV onto the global map
    x1 = max(0, int(tx))
    y1 = max(0, int(ty))
    x2 = min(global_size, int(tx + 250))
    y2 = min(global_size, int(ty + 250))

    # Calculate the corresponding region in the rotated BEV
    bev_x1 = max(0, -int(tx))
    bev_y1 = max(0, -int(ty))
    bev_x2 = bev_x1 + (x2 - x1)
    bev_y2 = bev_y1 + (y2 - y1)

    # Place the rotated BEV onto the global map
    if x1 < x2 and y1 < y2 and bev_x1 < bev_x2 and bev_y1 < bev_y2:
        canvas[y1:y2, x1:x2] = rotated_bev[bev_y1:bev_y2, bev_x1:bev_x2]

    # Create a mask for the valid pixels (non-zero alpha)
    mask = canvas[:, :, 3] > 50

    # Convert to BGR
    transformed_bev = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)

    return transformed_bev, mask


def draw_trajectory(map_image, trajectory, maze_size=15, global_size=500):
    """
    Draws the agent trajectory on the map with corrected coordinates.
    """
    if not trajectory:
        return

    # Scale factor from maze coordinates to global map coordinates
    scale = global_size / maze_size

    # CORRECTED: Convert trajectory to pixel coordinates without y-flip
    pixel_trajectory = []
    for x, y in trajectory:
        pixel_x = int(x * scale)
        pixel_y = int(y * scale)
        pixel_trajectory.append((pixel_x, pixel_y))

    # Draw the trajectory line
    for i in range(1, len(pixel_trajectory)):
        cv2.line(map_image, pixel_trajectory[i - 1], pixel_trajectory[i], (0, 0, 255), 2)

    # Draw start and end points
    cv2.circle(map_image, pixel_trajectory[0], 5, (0, 255, 0), -1)  # Green start
    cv2.circle(map_image, pixel_trajectory[-1], 5, (255, 0, 0), -1)  # Red end


def create_slam_video(output_dir, num_frames, fps=10):
    """
    Creates a video from the SLAM map frames.
    """
    frames = []
    frame_indices = list(range(0, num_frames, 10)) + [num_frames - 1]
    if not frame_indices:
        frame_indices = [0]

    for i in sorted(frame_indices):
        frame_path = os.path.join(output_dir, f'slam_map_frame_{i:03d}.png')
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)

    if not frames:
        print("No frames found for video creation.")
        return

    output_path = os.path.join(output_dir, 'slam_reconstruction.mp4')

    # Create video writer
    height, width = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # Write frames
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")


def run_visualization_pipeline(dataset_path, output_dir, model_path=None, max_frames=1000, frame_interval=5):
    """
    Runs the full visualization pipeline with sparse frame generation.
    
    Args:
        dataset_path (str): Path to the dataset
        output_dir (str): Path to save results
        model_path (str, optional): Path to the BEV model
        max_frames (int, optional): Maximum number of frames to process
        frame_interval (int, optional): Interval between saved mapped frames
    """
    import os
    import time
    
    start_time = time.time()
    
    # Create output directories
    bev_mapping_dir = os.path.join(output_dir, 'bev_mapping')
    slam_dir = os.path.join(output_dir, 'slam_reconstruction')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(bev_mapping_dir, exist_ok=True)
    os.makedirs(slam_dir, exist_ok=True)
    
    # Run the BEV mapping visualization with sparse frame generation
    print("\n--- Running BEV Mapping Visualization ---")
    visualize_agent_bev_mapping(dataset_path, bev_mapping_dir, model_path, 
                               max_frames=max_frames, frame_interval=frame_interval)
    
    # Run the SLAM reconstruction using the sparse frames
    print("\n--- Running SLAM Reconstruction ---")
    visualize_slam_reconstruction(dataset_path, slam_dir, model_path, max_frames=max_frames)
    
    elapsed = time.time() - start_time
    print(f"\nVisualization pipeline completed successfully in {elapsed:.2f} seconds!")
    print(f"All results saved to {output_dir}")