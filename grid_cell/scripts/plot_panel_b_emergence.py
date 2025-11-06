#!/usr/bin/env python3
"""Generate Nature-style Panel B showing emergence of grid-like firing fields."""

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

from grid_cell_visualization import GridCellVisualizer, load_model, load_dataloader


def clone_batches(dataloader, num_batches: int) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    for idx, batch in enumerate(dataloader):
        if idx >= num_batches:
            break
        cloned: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                cloned[key] = value.clone().cpu()
            else:
                cloned[key] = value
        batches.append(cloned)
    if not batches:
        raise RuntimeError("No batches collected for visualization; increase num_batches or check dataloader.")
    return batches


def collect_activations(
    model: torch.nn.Module,
    batches: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    positions_store: List[np.ndarray] = []
    activations_store: List[np.ndarray] = []

    with torch.no_grad():
        for batch in batches:
            positions = batch["positions"].to(device)
            angles = batch["angles"].to(device)
            velocities = batch["velocities"].to(device)
            ang_vels = batch["angular_velocities"].to(device)

            velocity_input = torch.cat([velocities, ang_vels.unsqueeze(-1)], dim=-1)
            init_pos = positions[:, 0]
            init_hd = angles[:, 0]

            outputs = model(velocity_input, init_pos, init_hd)

            # Drop the initial frame (t = 0) for analysis consistency.
            pos_np = positions[:, 1:].contiguous().view(-1, 2).cpu().numpy()
            bottleneck = outputs["bottleneck"][:, 1:].contiguous()
            activ_np = bottleneck.view(-1, bottleneck.shape[-1]).cpu().numpy()

            positions_store.append(pos_np)
            activations_store.append(activ_np)

    all_positions = np.concatenate(positions_store, axis=0)
    all_activations = np.concatenate(activations_store, axis=0)
    return all_positions, all_activations


def compute_stage_maps(
    visualizer: GridCellVisualizer,
    positions: np.ndarray,
    activations: np.ndarray,
) -> Dict[int, Dict[str, np.ndarray]]:
    scorer = visualizer.scorer
    num_units = activations.shape[1]
    maps: Dict[int, Dict[str, np.ndarray]] = {}

    for idx in range(num_units):
        rate_map = scorer.calculate_ratemap(
            positions[:, 0],
            positions[:, 1],
            activations[:, idx],
        )
        rate_map = np.nan_to_num(rate_map, nan=0.0)
        smooth_map = gaussian_filter(rate_map, sigma=1.0)
        score60, score90, mask60, mask90, sac = scorer.get_scores(rate_map)

        maps[idx] = {
            "ratemap": rate_map,
            "ratemap_smooth": smooth_map,
            "sac": sac,
            "gridness60": float(score60),
            "gridness90": float(score90),
            "mask60": mask60,
            "mask90": mask90,
        }

    return maps


def select_neurons_with_gain(
    before_maps: Dict[int, Dict[str, np.ndarray]],
    after_maps: Dict[int, Dict[str, np.ndarray]],
    desired: int,
    min_after: float,
    min_delta: float,
) -> np.ndarray:
    deltas: List[Tuple[float, float, float, int]] = []
    for idx in sorted(before_maps.keys()):
        before_score = before_maps[idx]["gridness60"]
        after_score = after_maps[idx]["gridness60"]
        delta = after_score - before_score
        deltas.append((delta, after_score, before_score, idx))

    # Filter using thresholds first
    filtered = [item for item in deltas if item[0] >= min_delta and item[1] >= min_after]
    if filtered:
        filtered.sort(reverse=True)
        take = min(desired, len(filtered))
        return np.array([item[3] for item in filtered[:take]], dtype=int)

    # Otherwise fall back to pure top-difference selection.
    deltas.sort(reverse=True)
    take = min(desired, len(deltas))
    return np.array([item[3] for item in deltas[:take]], dtype=int)


def save_array_csv(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, array, delimiter=",", fmt="%.6f")


def save_gridness_summary(path: str, entries: List[Tuple[int, str, float, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = "neuron_idx,stage,gridness_60,gridness_90"
    lines = [header]
    for idx, stage, s60, s90 in entries:
        lines.append(f"{idx},{stage},{s60:.6f},{s90:.6f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_panel_b(
    visualizer: GridCellVisualizer,
    before_maps: Dict[int, Dict[str, np.ndarray]],
    after_maps: Dict[int, Dict[str, np.ndarray]],
    neuron_indices: Sequence[int],
    output_base: str,
    env_size: float,
    before_label: str,
    after_label: str,
) -> None:
    nature_rc = {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }

    num_cols = len(neuron_indices)

    all_ratemaps = []
    for idx in neuron_indices:
        all_ratemaps.append(before_maps[idx]["ratemap_smooth"])
        all_ratemaps.append(after_maps[idx]["ratemap_smooth"])
    concat = np.concatenate([m.reshape(-1) for m in all_ratemaps])
    vmax = np.percentile(concat, 99.0)
    vmin = 0.0

    with plt.rc_context(nature_rc):
        fig = plt.figure(figsize=(num_cols * 3.3, 8.6))
        gs = gridspec.GridSpec(4, num_cols, height_ratios=[1, 1, 1, 1], hspace=0.08, wspace=0.1)

        for col, neuron_idx in enumerate(neuron_indices):
            # Before training ratemap
            ax_rm_before = fig.add_subplot(gs[0, col])
            im = ax_rm_before.imshow(
                before_maps[neuron_idx]["ratemap_smooth"],
                origin="lower",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                extent=[0, env_size, 0, env_size],
            )
            ax_rm_before.set_xticks([])
            ax_rm_before.set_yticks([])
            ax_rm_before.set_title(f"Neuron {neuron_idx}", fontweight="bold")
            ax_rm_before.text(
                0.02,
                0.92,
                f"Gridness {before_maps[neuron_idx]['gridness60']:.2f}",
                transform=ax_rm_before.transAxes,
                color="#1f1f1f",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
            )

            # Before training SAC
            ax_sac_before = fig.add_subplot(gs[1, col])
            sac_img = ax_sac_before.imshow(
                before_maps[neuron_idx]["sac"],
                origin="lower",
                cmap=visualizer.autocorr_cmap,
                vmin=-1,
                vmax=1,
            )
            ax_sac_before.set_xticks([])
            ax_sac_before.set_yticks([])

            mask_min, mask_max = before_maps[neuron_idx]["mask60"]
            center = visualizer.scorer._nbins - 1
            for radius in (mask_min, mask_max):
                ax_sac_before.add_artist(
                    Circle((center, center), radius * visualizer.scorer._nbins, fill=False, color="black", linewidth=0.8)
                )

            # After training ratemap
            ax_rm_after = fig.add_subplot(gs[2, col])
            ax_rm_after.imshow(
                after_maps[neuron_idx]["ratemap_smooth"],
                origin="lower",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                extent=[0, env_size, 0, env_size],
            )
            ax_rm_after.set_xticks([])
            ax_rm_after.set_yticks([])
            ax_rm_after.text(
                0.02,
                0.92,
                f"Gridness {after_maps[neuron_idx]['gridness60']:.2f}",
                transform=ax_rm_after.transAxes,
                color="#1f1f1f",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
            )

            # After training SAC
            ax_sac_after = fig.add_subplot(gs[3, col])
            ax_sac_after.imshow(
                after_maps[neuron_idx]["sac"],
                origin="lower",
                cmap=visualizer.autocorr_cmap,
                vmin=-1,
                vmax=1,
            )
            ax_sac_after.set_xticks([])
            ax_sac_after.set_yticks([])

            mask_min_a, mask_max_a = after_maps[neuron_idx]["mask60"]
            center_a = visualizer.scorer._nbins - 1
            for radius in (mask_min_a, mask_max_a):
                ax_sac_after.add_artist(
                Circle((center_a, center_a), radius * visualizer.scorer._nbins, fill=False, color="black", linewidth=0.8)
                )

        # Row labels for before/after sections
        fig.text(0.02, 0.70, before_label, fontsize=12, fontweight="bold", va="center", ha="left")
        fig.text(0.02, 0.23, after_label, fontsize=12, fontweight="bold", va="center", ha="left")

        cbar_ax1 = fig.add_axes([0.93, 0.56, 0.015, 0.32])
        fig.colorbar(im, cax=cbar_ax1, label="Firing rate (a.u.)")
        cbar_ax2 = fig.add_axes([0.93, 0.14, 0.015, 0.32])
        fig.colorbar(sac_img, cax=cbar_ax2, label="Spatial autocorr.")

        # Add minimal panel labels
        fig.text(0.02, 0.94, "Panel B", fontsize=13, fontweight="bold")
        fig.text(0.5, 0.94, "Figure S1-B: Emergence of Spatially-Tuned Firing Fields", fontsize=14, fontweight="bold", ha="center")

        fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{output_base}.svg", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Panel B emergence figure.")
    parser.add_argument("--before_checkpoint", required=True, help="Path to early training model checkpoint.")
    parser.add_argument("--after_checkpoint", required=True, help="Path to converged model checkpoint.")
    parser.add_argument("--output_dir", default="supplementary_figure/S-fig1", help="Directory to store outputs.")
    parser.add_argument("--figure_base", default="figure_s1b_emergence", help="Base filename for saved artifacts.")
    parser.add_argument("--suffix", default="nature", help="Filename suffix for disambiguation.")
    parser.add_argument("--num_neurons", type=int, default=4, help="Number of representative neurons to plot.")
    parser.add_argument("--num_batches", type=int, default=25, help="How many batches to process for activations.")
    parser.add_argument("--batch_size", type=int, default=32, help="Dataloader batch size.")
    parser.add_argument("--device", default="cuda:0", help="Device for model inference.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to use.")
    parser.add_argument("--nbins", type=int, default=40, help="Resolution of rate maps.")
    parser.add_argument("--env_size", type=float, default=15.0, help="Environment size for extent annotations.")
    parser.add_argument("--min_after_gridness", type=float, default=0.0, help="Minimum post-training gridness required for selection.")
    parser.add_argument("--min_delta_gridness", type=float, default=0.5, help="Minimum improvement in gridness required for selection.")
    parser.add_argument("--before_caption", default="b  Before Training", help="Row label for the pre-training panels.")
    parser.add_argument("--after_caption", default="c  After Training", help="Row label for the post-training panels.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_base = os.path.join(args.output_dir, f"{args.figure_base}_{args.suffix}")
    data_dir = os.path.join(args.output_dir, f"{args.figure_base}_{args.suffix}_data")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataloader = load_dataloader(split=args.split, batch_size=args.batch_size)
    if dataloader is None:
        raise RuntimeError("Failed to create dataloader; cannot proceed.")

    batches = clone_batches(dataloader, args.num_batches)

    visualizer = GridCellVisualizer(args.output_dir, nbins=args.nbins, env_size=args.env_size)

    model_before = load_model(args.before_checkpoint, device)
    if model_before is None:
        raise RuntimeError("Failed to load before-training checkpoint.")
    before_positions, before_activations = collect_activations(model_before, batches, device)

    model_after = load_model(args.after_checkpoint, device)
    if model_after is None:
        raise RuntimeError("Failed to load after-training checkpoint.")
    after_positions, after_activations = collect_activations(model_after, batches, device)

    before_maps_all = compute_stage_maps(visualizer, before_positions, before_activations)
    after_maps_all = compute_stage_maps(visualizer, after_positions, after_activations)

    top_indices = select_neurons_with_gain(
        before_maps_all,
        after_maps_all,
        args.num_neurons,
        args.min_after_gridness,
        args.min_delta_gridness,
    )

    before_maps = {idx: before_maps_all[int(idx)] for idx in top_indices}
    after_maps = {idx: after_maps_all[int(idx)] for idx in top_indices}

    print("Selected neurons (sorted by 60° gridness gain):", [int(i) for i in top_indices])

    entries: List[Tuple[int, str, float, float]] = []
    for idx in top_indices:
        entries.append((int(idx), "before", before_maps[int(idx)]["gridness60"], before_maps[int(idx)]["gridness90"]))
        entries.append((int(idx), "after", after_maps[int(idx)]["gridness60"], after_maps[int(idx)]["gridness90"]))

        save_array_csv(
            os.path.join(data_dir, f"neuron_{int(idx):03d}_before_ratemap.csv"),
            before_maps[int(idx)]["ratemap"],
        )
        save_array_csv(
            os.path.join(data_dir, f"neuron_{int(idx):03d}_before_sac.csv"),
            before_maps[int(idx)]["sac"],
        )
        save_array_csv(
            os.path.join(data_dir, f"neuron_{int(idx):03d}_after_ratemap.csv"),
            after_maps[int(idx)]["ratemap"],
        )
        save_array_csv(
            os.path.join(data_dir, f"neuron_{int(idx):03d}_after_sac.csv"),
            after_maps[int(idx)]["sac"],
        )

    save_gridness_summary(
        os.path.join(data_dir, "gridness_summary.csv"),
        entries,
    )

    plot_panel_b(
        visualizer,
        before_maps,
        after_maps,
        top_indices,
        output_base,
        args.env_size,
        args.before_caption,
        args.after_caption,
    )

    print("Artifacts saved:")
    print(f"  PNG: {output_base}.png")
    print(f"  SVG: {output_base}.svg")
    print(f"  Data directory: {data_dir}")


if __name__ == "__main__":
    main()
