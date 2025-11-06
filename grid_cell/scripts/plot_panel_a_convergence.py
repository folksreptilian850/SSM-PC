#!/usr/bin/env python3
"""Generate Panel A convergence figure from a training checkpoint."""

import argparse
import os
from typing import Dict, List

import torch

from utils.visualization import NavigationVisualizer


def _sanitize_metrics(metrics: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Ensure metrics are plain Python floats for downstream plotting."""
    sanitized = {}
    for key, values in metrics.items():
        sanitized[key] = [float(v) for v in values]
    return sanitized


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Nature-style convergence curves (Panel A).")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a training checkpoint that contains metrics_history.",
    )
    parser.add_argument(
        "--output_dir",
        default="supplementary_figure",
        help="Directory where the figure will be saved.",
    )
    parser.add_argument(
        "--suffix",
        default="nature",
        help="Suffix appended to the output filename for disambiguation.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Override the epoch number stamped on the figure (defaults to checkpoint epoch).",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "metrics_history" not in checkpoint:
        raise KeyError("Checkpoint does not contain metrics_history; cannot plot convergence curves.")

    metrics_history = _sanitize_metrics(checkpoint["metrics_history"])
    plotted_epoch = args.epoch or checkpoint.get("epoch") or len(metrics_history.get("train_total", []))

    os.makedirs(args.output_dir, exist_ok=True)
    visualizer = NavigationVisualizer(args.output_dir)
    visualizer.plot_metrics(metrics_history, plotted_epoch, suffix=args.suffix)

    suffix_clean = f"_{args.suffix}" if args.suffix else ""
    base = f"figure_s1a_convergence_epoch_{plotted_epoch}{suffix_clean}"
    print("Artifacts saved:")
    print(f"  PNG: {os.path.join(args.output_dir, base + '.png')}")
    print(f"  SVG: {os.path.join(args.output_dir, base + '.svg')}")
    print(f"  CSV (total loss): {os.path.join(args.output_dir, base + '_total_loss.csv')}")
    print(f"  CSV (place loss): {os.path.join(args.output_dir, base + '_place_loss.csv')}")
    print(f"  CSV (head direction loss): {os.path.join(args.output_dir, base + '_head_direction_loss.csv')}")


if __name__ == "__main__":
    main()
