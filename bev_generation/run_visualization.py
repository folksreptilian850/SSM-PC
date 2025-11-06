#run_visualization.py
import os
import argparse
from bev_mapping import run_visualization_pipeline


def main():
    parser = argparse.ArgumentParser(description='BEV Mapping and SLAM Reconstruction')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                        help='Directory to save the output')
    parser.add_argument('--model_path', type=str, default=os.environ.get("BEV_MODEL_PATH"),
                        help='Path to the BEV prediction model checkpoint (or set BEV_MODEL_PATH env var)')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Maximum number of frames to process')
    parser.add_argument('--mode', type=str, choices=['all', 'mapping', 'slam'], default='all',
                        help='Which visualization to run')

    args = parser.parse_args()

    if not args.model_path:
        parser.error("Please provide --model_path or set the BEV_MODEL_PATH environment variable.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'all':
        # Run the full pipeline
        run_visualization_pipeline(
            args.dataset_path,
            args.output_dir,
            args.model_path,
            args.max_frames,
            frame_interval=10
        )
    elif args.mode == 'mapping':
        # Run only the BEV mapping visualization
        from bev_mapping import visualize_agent_bev_mapping
        mapping_dir = os.path.join(args.output_dir, 'bev_mapping')
        os.makedirs(mapping_dir, exist_ok=True)
        visualize_agent_bev_mapping(args.dataset_path, mapping_dir, args.model_path)
    elif args.mode == 'slam':
        # Run only the SLAM reconstruction
        from bev_mapping import visualize_slam_reconstruction
        slam_dir = os.path.join(args.output_dir, 'slam_reconstruction')
        os.makedirs(slam_dir, exist_ok=True)
        visualize_slam_reconstruction(
            args.dataset_path,
            slam_dir,
            args.model_path,
            args.max_frames
        )


if __name__ == "__main__":
    main()
