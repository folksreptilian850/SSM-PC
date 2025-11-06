#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from models.place_hd_cells import HeadDirectionCellEnsemble, PlaceCellEnsemble
from models.toroidal_grid_cell import GridCellNetwork
from vis_cell import (
    build_and_load_model,
    decode_predictions,
    load_single_trajectory,
    prepare_inference_input,
)


def resolve_font(preferred_family: str, font_path: Path = None, fallback_family: str = "Arial") -> str:
    """Resolve a usable font family, mimicking the nature-style setup."""
    if font_path:
        font_path = font_path.expanduser().resolve()
        if not font_path.exists():
            raise FileNotFoundError(f"Font path does not exist: {font_path}")
        fm.fontManager.addfont(str(font_path))
        fm._rebuild()
        family = preferred_family or font_path.stem
        return family

    try:
        fm.findfont(preferred_family, fallback_to_default=False)
        return preferred_family
    except Exception:
        pass

    try:
        import subprocess

        result = subprocess.run(
            ["fc-match", "-f", "%{file}\n", preferred_family],
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = Path(result.stdout.strip())
        if candidate.exists():
            fm.fontManager.addfont(str(candidate))
            fm._rebuild()
            fm.findfont(preferred_family, fallback_to_default=False)
            return preferred_family
    except Exception:
        pass

    fm.findfont(fallback_family, fallback_to_default=True)
    return fallback_family


def compute_errors(
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    gt_angles: np.ndarray,
    pred_angles: np.ndarray,
) -> Tuple[float, float]:
    """Compute mean positional and rotational errors excluding t=0."""
    if gt_positions.shape[0] <= 1:
        return 0.0, 0.0
    pos_errors = np.linalg.norm(gt_positions[1:] - pred_positions[1:], axis=1)
    mean_pos_error = float(pos_errors.mean())
    angle_diffs = np.angle(np.exp(1j * (gt_angles[1:] - pred_angles[1:])))
    mean_angle_error = float(np.abs(angle_diffs).mean())
    return mean_pos_error, math.degrees(mean_angle_error)


def figure_to_array(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib figure to a numpy array."""
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((h, w, 3))


def build_random_model(config: Config):
    """Construct a grid cell model with random initialization (no checkpoint load)."""
    place_cells = PlaceCellEnsemble(
        n_cells=config.PLACE_CELLS_N,
        scale=config.PLACE_CELLS_SCALE,
        pos_min=0,
        pos_max=config.ENV_SIZE,
        seed=config.SEED,
    )
    hd_cells = HeadDirectionCellEnsemble(
        n_cells=config.HD_CELLS_N,
        concentration=config.HD_CELLS_CONCENTRATION,
        seed=config.SEED,
    )
    model = GridCellNetwork(
        place_cells=place_cells,
        hd_cells=hd_cells,
        input_size=3,
        hidden_size=config.HIDDEN_SIZE,
        bottleneck_size=256,
        dropout_rate=config.DROPOUT_RATE,
    )
    model.eval()
    return model, place_cells, hd_cells


def render_frame(
    seq_len: int,
    panel_infos: List[Dict[str, np.ndarray]],
    gt_positions: np.ndarray,
    gt_angles: np.ndarray,
    maze_size: float,
    dpi: int,
) -> np.ndarray:
    """Render one frame showing trajectory comparisons for all requested panels."""
    width, height = 1920, 1080
    fig, axes = plt.subplots(
        1,
        len(panel_infos),
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.08, wspace=0.25)
    fig.suptitle(f"Sequence Length: {seq_len}", fontsize=28, fontweight="bold", y=0.94)

    colors = {"gt": "#1f77b4", "pred": "#d62728"}

    for ax, panel in zip(axes, panel_infos):
        pred_data = panel["prediction"]
        pred_positions = pred_data["positions"][:seq_len]
        pred_angles = pred_data["angles"][:seq_len]
        gt_pos_slice = gt_positions[:seq_len]
        gt_ang_slice = gt_angles[:seq_len]

        ax.plot(
            gt_pos_slice[:, 0],
            gt_pos_slice[:, 1],
            color=colors["gt"],
            linewidth=2.0,
            label="Ground Truth",
        )
        ax.plot(
            pred_positions[:, 0],
            pred_positions[:, 1],
            color=colors["pred"],
            linewidth=2.0,
            linestyle="--",
            label="Prediction",
        )

        ax.scatter(
            gt_pos_slice[0, 0],
            gt_pos_slice[0, 1],
            color="#2ca02c",
            s=60,
            zorder=5,
            label="Start",
        )
        ax.scatter(
            gt_pos_slice[-1, 0],
            gt_pos_slice[-1, 1],
            color=colors["gt"],
            s=70,
            marker="s",
            zorder=6,
            label="GT End",
        )
        ax.scatter(
            pred_positions[-1, 0],
            pred_positions[-1, 1],
            color=colors["pred"],
            s=70,
            marker="^",
            zorder=6,
            label="Pred End",
        )

        ax.set_xlim(0, maze_size)
        ax.set_ylim(0, maze_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.linspace(0, maze_size, 4))
        ax.set_yticks(np.linspace(0, maze_size, 4))
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_title(panel["title"], fontsize=22, pad=18)
        ax.tick_params(axis="both", labelsize=12)

        mean_pos_err, mean_ang_err = compute_errors(gt_pos_slice, pred_positions, gt_ang_slice, pred_angles)
        text = f"mean Δpos: {mean_pos_err:.2f} m\nmean Δrot: {mean_ang_err:.1f}°"
        ax.text(
            0.03,
            0.96,
            text,
            transform=ax.transAxes,
            fontsize=14,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.6, edgecolor="none"),
        )

        ax.legend(loc="lower right", fontsize=11, frameon=False)

    frame = figure_to_array(fig)
    plt.close(fig)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-epoch trajectory comparison video.")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing epoch_NNN.pth checkpoints.",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=Path,
        required=True,
        help="Directory with trajectory data (frame_info.json and frames).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("epoch_trajectory_comparison.mp4"),
        help="Output video path.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[0, 5, 50, 150],
        help="Epoch numbers to visualize.",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=2,
        help="Minimum sequence length (inclusive).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=100,
        help="Maximum sequence length (inclusive).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--font_family",
        type=str,
        default="Helvetica",
        help="Preferred font family for titles and annotations.",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default=None,
        help="Optional explicit font file to load.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Matplotlib DPI controlling the render resolution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config()
    checkpoint_dir = args.checkpoint_dir.resolve()
    trajectory_dir = args.trajectory_dir.resolve()
    output_path = args.output_path.resolve()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")

    font_family = resolve_font(args.font_family, Path(args.font_path) if args.font_path else None)
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.sans-serif"] = [font_family, "Arial", "Liberation Sans", "DejaVu Sans"]

    max_seq_len = args.max_seq_len
    if args.min_seq_len < 1 or max_seq_len < args.min_seq_len:
        raise ValueError("Sequence length bounds are invalid.")

    positions, angles, velocities, angular_vels = load_single_trajectory(
        str(trajectory_dir),
        sequence_length=max_seq_len,
        stride=1,
    )

    velocity_input, init_pos, init_hd = prepare_inference_input(positions, angles, velocities, angular_vels)
    gt_positions = positions.cpu().numpy()
    gt_angles = angles.cpu().numpy()

    panel_infos: List[Dict[str, np.ndarray]] = []
    place_cells_ref = None
    hd_cells_ref = None

    available_epoch_nums = sorted(
        int(p.stem.split("_")[1])
        for p in checkpoint_dir.glob("epoch_*.pth")
        if p.stem.split("_")[1].isdigit()
    )

    for requested_epoch in args.epochs:
        ckpt_path = checkpoint_dir / f"epoch_{requested_epoch}.pth"
        actual_epoch = requested_epoch
        random_init = False

        if not ckpt_path.exists():
            if requested_epoch == 0:
                print("[info] Requested epoch 0 missing; using randomly initialized model.")
                model, place_cells, hd_cells = build_random_model(config)
                random_init = True
            else:
                if available_epoch_nums:
                    nearest_epoch = min(available_epoch_nums, key=lambda e: abs(e - requested_epoch))
                    ckpt_path = checkpoint_dir / f"epoch_{nearest_epoch}.pth"
                    actual_epoch = nearest_epoch
                    print(
                        f"[warn] Requested epoch {requested_epoch} missing. "
                        f"Using nearest available epoch {actual_epoch}."
                    )
                else:
                    print(f"[warn] No checkpoints available in {checkpoint_dir}.")
                    continue

        if not random_init:
            if not ckpt_path.exists():
                print(f"[warn] Checkpoint not found: {ckpt_path}")
                continue
            model, place_cells, hd_cells = build_and_load_model(str(ckpt_path), config)

        if place_cells_ref is None:
            place_cells_ref = place_cells
            hd_cells_ref = hd_cells
        model.eval()
        with torch.no_grad():
            outputs = model(velocity_input, init_pos, init_hd)
            place_logits = outputs["place_logits"].squeeze(0)
            hd_logits = outputs["hd_logits"].squeeze(0)
            pred_positions, pred_angles = decode_predictions(place_logits, hd_logits, place_cells_ref, hd_cells_ref)
        pred_positions = pred_positions.cpu().numpy()
        pred_angles = pred_angles.cpu().numpy()
        pred_positions[0] = gt_positions[0]
        pred_angles[0] = gt_angles[0]

        if random_init:
            title = "Epoch 0 (Random Init)"
        elif actual_epoch != requested_epoch:
            title = f"Req {requested_epoch} → Epoch {actual_epoch}"
        else:
            title = f"Epoch {actual_epoch}"

        panel_infos.append(
            {
                "requested_epoch": requested_epoch,
                "actual_epoch": actual_epoch,
                "title": title,
                "prediction": {
                    "positions": pred_positions,
                    "angles": pred_angles,
                },
            }
        )

    if not panel_infos:
        raise RuntimeError("No checkpoints were loaded successfully. Please verify the epoch list.")

    min_seq_len = max(args.min_seq_len, 2)
    total_frames = max_seq_len - min_seq_len + 1
    panel_titles = [panel["title"] for panel in panel_infos]
    print(f"Rendering {total_frames} frames for panels: {panel_titles}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for seq_len in range(min_seq_len, max_seq_len + 1):
            frame_rgb = render_frame(
                seq_len=seq_len,
                panel_infos=panel_infos,
                gt_positions=gt_positions,
                gt_angles=gt_angles,
                maze_size=config.ENV_SIZE,
                dpi=args.dpi,
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            if seq_len % 10 == 0 or seq_len == min_seq_len or seq_len == max_seq_len:
                print(f"[info] Rendered sequence length {seq_len}")
    finally:
        writer.release()

    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
