#!/usr/bin/env python

import os
import sys
import subprocess
from datetime import datetime
from typing import Iterable, List


def find_available_log_dirs(verbose: bool = True) -> List[str]:
    """Return sorted log directories that contain checkpoint files."""
    log_base_dir = "log_train_grid_cells"

    if not os.path.exists(log_base_dir):
        if verbose:
            print(f"[warn] Log base directory not found: {log_base_dir}")
        return []

    log_dirs: List[str] = []
    for item in sorted(os.listdir(log_base_dir)):
        item_path = os.path.join(log_base_dir, item)
        if not os.path.isdir(item_path):
            continue

        checkpoints_dir = os.path.join(item_path, "checkpoints")
        if not os.path.isdir(checkpoints_dir):
            continue

        checkpoint_files = [
            f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")
        ]
        if checkpoint_files:
            log_dirs.append(item_path)
            if verbose:
                print(
                    f"[info] Found log directory {item_path} "
                    f"({len(checkpoint_files)} checkpoints)"
                )

    return log_dirs


def run_gridness_correlation(
    log_dirs: Iterable[str],
    save_dir: str,
    device: str,
    max_checkpoints: int,
    num_batches_gridness: int,
    num_trajectories_performance: int,
) -> bool:
    """Invoke the correlation analysis script with the provided options."""
    cmd = [
        "python",
        "gridness_performance_correlation.py",
        "--log_dirs",
        *log_dirs,
        "--save_dir",
        save_dir,
        "--device",
        device,
        "--max_checkpoints",
        str(max_checkpoints),
        "--num_batches_gridness",
        str(num_batches_gridness),
        "--num_trajectories_performance",
        str(num_trajectories_performance),
    ]

    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as exc:
        print(f"[error] Analysis command failed: {exc}")
        return False
    except KeyboardInterrupt:
        print("[info] Analysis interrupted by user.")
        return False

    return True


def run_analysis_with_recommended_settings() -> bool:
    print("Scanning for training logs...")
    log_dirs = find_available_log_dirs()
    if not log_dirs:
        print("No training logs with checkpoints were found.")
        print("Expected structure: log_train_grid_cells/*/checkpoints/*.pth")
        return False

    print("\nAvailable log directories:")
    for idx, log_dir in enumerate(log_dirs, 1):
        print(f"  {idx}. {log_dir}")

    if len(log_dirs) == 1:
        selected_dirs = log_dirs
        print(f"\nUsing the only available directory: {log_dirs[0]}")
    else:
        print(
            "\nSelect directories to analyse "
            "(indices separated by spaces, press Enter for all):"
        )
        try:
            user_input = input().strip()
        except KeyboardInterrupt:
            print("\nSelection cancelled by user.")
            return False

        if not user_input:
            selected_dirs = log_dirs
            print("Using all directories.")
        else:
            try:
                indices = [int(x) - 1 for x in user_input.split()]
                selected_dirs = [
                    log_dirs[i] for i in indices if 0 <= i < len(log_dirs)
                ]
            except ValueError:
                selected_dirs = []

            if selected_dirs:
                print(f"Selected {len(selected_dirs)} directories.")
            else:
                print("Invalid selection; defaulting to all directories.")
                selected_dirs = log_dirs

    if not selected_dirs:
        print("No valid directories selected.")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = f"gridness_analysis_{timestamp}"

    print("\nLaunching analysis...")
    print(f"Output directory: {analysis_dir}")
    print("=" * 60)

    success = run_gridness_correlation(
        log_dirs=selected_dirs,
        save_dir=analysis_dir,
        device="cuda:0",
        max_checkpoints=8,
        num_batches_gridness=15,
        num_trajectories_performance=8,
    )

    if not success:
        return False

    print("\nAnalysis completed successfully.")
    print(f"Results saved to: {analysis_dir}")

    expected_files = [
        "correlation_report.md",
        "gridness_performance_correlation.png",
        "Figure_2b_gridness_correlation.png",
        "intermediate_results.json",
    ]

    print("\nGenerated files:")
    for filename in expected_files:
        filepath = os.path.join(analysis_dir, filename)
        status = "present" if os.path.exists(filepath) else "missing"
        print(f"  {filename}: {status}")

    return True


def run_quick_test() -> bool:
    print("Running quick-test mode...")
    log_dirs = find_available_log_dirs(verbose=False)
    if not log_dirs:
        print("No training logs with checkpoints were found.")
        return False

    test_log_dir = log_dirs[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"gridness_test_{timestamp}"

    print(f"Using log directory: {test_log_dir}")
    print(f"Output directory: {test_dir}")
    print("=" * 60)

    return run_gridness_correlation(
        log_dirs=[test_log_dir],
        save_dir=test_dir,
        device="cuda:0",
        max_checkpoints=3,
        num_batches_gridness=5,
        num_trajectories_performance=3,
    )


def main() -> None:
    print("Gridness versus path-integration performance analysis")
    print("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_quick_test()
    else:
        print("Select mode: [1] full analysis (default)  [2] quick test")
        try:
            choice = input("Enter 1 or 2: ").strip()
        except KeyboardInterrupt:
            print("\nCancelled by user.")
            return

        success = run_quick_test() if choice == "2" else run_analysis_with_recommended_settings()

    if success:
        print("\nAnalysis finished.")
        print("Review correlation_report.md for details.")
        print("Figure_2b_gridness_correlation.png is export-ready.")
        print(
            "If correlations look weak, consider analysing more checkpoints "
            "or adjusting evaluation parameters."
        )
    else:
        print("\nAnalysis did not complete successfully.")


if __name__ == "__main__":
    main()
