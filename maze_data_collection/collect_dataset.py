from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List

DEFAULT_ENV = "memory_maze:MemoryMaze-15x15-ExtraObs"
DEFAULT_OUTPUT_ROOT = Path("datasets")


def build_command(args: argparse.Namespace, seed: int, dataset_dir: Path) -> List[str]:
    cmd = [
        "python",
        str(Path(__file__).parent / "gui" / "headless_collector.py"),
        "--env",
        args.env,
        "--seed",
        str(seed),
        "--collect_data",
        "--fov_angle",
        str(args.fov_angle),
        "--fps",
        str(args.fps),
        "--reset_steps",
        str(args.reset_steps),
        "--auto_nav",
        "--max_trajectories",
        str(args.max_trajectories),
        "--dataset_dir",
        str(dataset_dir),
    ]
    if args.visibility_range is not None:
        cmd.extend(["--visibility_range", str(args.visibility_range)])
    return cmd


def launch_collection(args: argparse.Namespace) -> None:
    seeds = [args.start_seed + i for i in range(args.num_processes)]
    args.output_root.mkdir(parents=True, exist_ok=True)

    processes = []
    for seed in seeds:
        dataset_dir = args.output_root / f"D{seed}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(args, seed, dataset_dir)
        print(f"Launching collection for seed {seed}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        processes.append(proc)

    for proc in processes:
        proc.wait()

    print("All collection processes completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch parallel maze data collection.")
    parser.add_argument("--env", default=DEFAULT_ENV, help="Gym environment identifier.")
    parser.add_argument("--start_seed", type=int, default=1, help="Starting seed value.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of parallel collectors.")
    parser.add_argument("--max_trajectories", type=int, default=500, help="Trajectories per dataset.")
    parser.add_argument("--fov_angle", type=float, default=75.0, help="Agent FOV angle.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering.")
    parser.add_argument("--reset_steps", type=int, default=500, help="Environment reset interval.")
    parser.add_argument("--visibility_range", type=float, default=None, help="Optional visibility mask range.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for collected datasets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch_collection(args)


if __name__ == "__main__":
    main()
