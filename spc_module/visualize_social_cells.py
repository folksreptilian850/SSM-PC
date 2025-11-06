import argparse
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from grid_cell.models.place_hd_cells import HeadDirectionCellEnsemble, PlaceCellEnsemble  # type: ignore
from spc_module.config import SPCConfig
from spc_module.social_grid_cell import SocialGridCellNetwork
from spc_module.trajectory import TrajectoryGenerator

plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "figure.titlesize": 16,
        "figure.dpi": 150,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


@dataclass
class ConditionAnalysis:
    peak_activation: float = 0.0
    activations: np.ndarray = field(default_factory=lambda: np.zeros(0))
    positions: Dict[str, np.ndarray] = field(default_factory=dict)
    plot_key: str = "self"


@dataclass
class NeuronAnalysisResult:
    neuron_idx: int
    cell_type: str = "Unclassified"
    condition_data: Dict[int, ConditionAnalysis] = field(default_factory=dict)


class CellTypeDetector:
    @staticmethod
    def classify(peaks: Dict[int, float]) -> str:
        high = 0.5
        low = 0.2
        responsive_self = peaks[1] > high
        responsive_peer = peaks[2] > high
        quiet_static = peaks[4] < low

        if not quiet_static:
            return "Other (Active when static)"
        if responsive_self and responsive_peer:
            return "Special SPC"
        if responsive_self and not responsive_peer:
            return "Pure Place Cell" if peaks[3] > high else "Mixed Response"
        if not responsive_self and responsive_peer:
            return "Pure SPC" if peaks[3] > high else "Mixed Response"
        return "Other"


PLOT_KEY_MAP = {
    "Pure SPC": {1: "self", 2: "peer", 3: "peer", 4: "peer"},
    "Special SPC": {1: "self", 2: "peer", 3: "self", 4: "peer"},
    "Pure Place Cell": {1: "self", 2: "peer", 3: "self", 4: "peer"},
    "Mixed Response": {1: "self", 2: "peer", 3: "self", 4: "peer"},
    "Other": {1: "self", 2: "peer", 3: "self", 4: "peer"},
    "Other (Active when static)": {1: "self", 2: "peer", 3: "self", 4: "peer"},
}


def occupancy_heatmap(positions: np.ndarray, activations: np.ndarray, grid_size: int, env_size: float, sigma: float,
                      vmin: float, vmax: float) -> np.ndarray:
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    counts = np.zeros_like(heatmap)
    clipped = np.clip(activations, vmin, vmax)
    scale = grid_size / env_size

    for traj_idx in range(positions.shape[0]):
        for step_idx in range(positions.shape[1]):
            x, y = positions[traj_idx, step_idx]
            gx = np.clip(int(x * scale), 0, grid_size - 1)
            gy = np.clip(int(y * scale), 0, grid_size - 1)
            heatmap[gy, gx] += clipped[traj_idx, step_idx]
            counts[gy, gx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        averaged = np.where(counts > 0, heatmap / counts, 0)
    return gaussian_filter(averaged, sigma=sigma)


def load_model(model_path: Path, device: torch.device) -> Tuple[SocialGridCellNetwork, SPCConfig]:
    config = SPCConfig()
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
    model_config = {
        "HIDDEN_SIZE": config.HIDDEN_SIZE,
        "LATENT_DIM": config.LATENT_DIM,
        "dropout_rate": config.DROPOUT_RATE,
        "ego_token_size": getattr(config, "ego_token_size", 4),
    }
    model = SocialGridCellNetwork(place_cells, hd_cells, model_config)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def collect_condition_data(model: SocialGridCellNetwork, config: SPCConfig, generator: TrajectoryGenerator,
                           device: torch.device, condition: int, num_reps: int) -> Dict[str, np.ndarray]:
    samples = generator.sample(condition, num_reps=num_reps)

    self_vel = torch.cat(
        [
            torch.from_numpy(samples["self_vel"]).float(),
            torch.from_numpy(samples["self_ang_vel"]).float().unsqueeze(-1),
        ],
        dim=-1,
    ).to(device)
    peer_vel = torch.cat(
        [
            torch.from_numpy(samples["peer_vel"]).float(),
            torch.from_numpy(samples["peer_ang_vel"]).float().unsqueeze(-1),
        ],
        dim=-1,
    ).to(device)
    self_init_pos = torch.from_numpy(samples["self_pos"][:, 0, :]).float().to(device)
    peer_init_pos = torch.from_numpy(samples["peer_pos"][:, 0, :]).float().to(device)
    self_init_hd = torch.from_numpy(np.arctan2(samples["self_vel"][:, 0, 1], samples["self_vel"][:, 0, 0])).float().to(device)
    peer_init_hd = torch.from_numpy(np.arctan2(samples["peer_vel"][:, 0, 1], samples["peer_vel"][:, 0, 0])).float().to(device)

    with torch.no_grad():
        outputs = model(self_vel, self_init_pos, self_init_hd, peer_vel, peer_init_pos, peer_init_hd)
    activations = outputs["bottleneck_self"].cpu().numpy()
    return {"positions": samples, "activations": activations}


def visualize_neuron(task: Tuple[NeuronAnalysisResult, Dict, str, float, float]) -> None:
    result, config_dict, output_dir, vmin, vmax = task
    config = config_dict

    titles = {
        1: "Self Moving / Peer Static",
        2: "Peer Moving / Self Static",
        3: "Both Moving",
        4: "Both Static",
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"Neuron {result.neuron_idx} – {result.cell_type}", fontsize=16, fontweight="bold", y=0.95)

    env_size = config["ENV_SIZE"]
    for idx, condition in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        data = result.condition_data[condition]
        key = PLOT_KEY_MAP[result.cell_type][condition]
        heatmap = occupancy_heatmap(
            data.positions[key],
            data.activations,
            grid_size=50,
            env_size=env_size,
            sigma=1.5,
            vmin=vmin,
            vmax=vmax,
        )

        image = ax.imshow(
            heatmap,
            cmap="jet",
            origin="lower",
            extent=[0, env_size, 0, env_size],
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )

        for trajectory in data.positions[key][: min(3, data.positions[key].shape[0])]:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="white", linewidth=2.5, alpha=0.9)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="black", linewidth=1.0, alpha=0.8)
        start = data.positions[key][0, 0]
        end = data.positions[key][0, -1]
        ax.scatter(start[0], start[1], color="lime", s=35, edgecolors="black", linewidth=1, label="Start")
        ax.scatter(end[0], end[1], color="red", s=35, edgecolors="black", linewidth=1, marker="s", label="End")

        ax.set_title(f"{titles[condition]}\nPeak: {data.peak_activation:.3f}", fontsize=12, pad=15)
        ax.set_xlabel("X (m)")
        if idx == 0:
            ax.set_ylabel("Y (m)")
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
        ax.set_xlim(0, env_size)
        ax.set_ylim(0, env_size)
        ax.set_xticks(np.linspace(0, env_size, 6))
        ax.set_yticks(np.linspace(0, env_size, 6))
        ax.grid(True, linestyle=":", alpha=0.6, color="white", linewidth=0.8)

    plt.subplots_adjust(left=0.06, bottom=0.15, right=0.88, top=0.85, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Activation", rotation=270, labelpad=18)

    cell_tag = result.cell_type.lower().replace(" ", "_").replace("(", "").replace(")", "")
    output_path = Path(output_dir) / f"neuron_{result.neuron_idx:03d}_{cell_tag}.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise social place-cell activations.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--output_dir", type=Path, default=Path("social_cell_viz"), help="Destination directory.")
    parser.add_argument("--num_reps", type=int, default=100, help="Trajectories per condition.")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2), help="Parallel processes for plotting.")
    parser.add_argument("--vmin", type=float, default=0.0, help="Minimum activation for colour mapping.")
    parser.add_argument("--vmax", type=float, default=0.8, help="Maximum activation for colour mapping.")
    args = parser.parse_args()

    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.model_path, device)
    generator = TrajectoryGenerator(config)
    num_neurons = config.LATENT_DIM

    results = {idx: NeuronAnalysisResult(neuron_idx=idx) for idx in range(num_neurons)}

    with torch.no_grad():
        for condition in tqdm(range(1, 5), desc="Conditions"):
            condition_data = collect_condition_data(model, config, generator, device, condition, args.num_reps)
            for neuron_idx in range(num_neurons):
                activations = condition_data["activations"][:, :, neuron_idx]
                results[neuron_idx].condition_data[condition] = ConditionAnalysis(
                    peak_activation=float(np.max(activations)),
                    activations=activations,
                    positions={"self": condition_data["positions"]["self_pos"], "peer": condition_data["positions"]["peer_pos"]},
                )

    categories = defaultdict(list)
    for neuron_idx in range(num_neurons):
        entry = results[neuron_idx]
        peaks = {cond: data.peak_activation for cond, data in entry.condition_data.items()}
        entry.cell_type = CellTypeDetector.classify(peaks)
        for cond, data in entry.condition_data.items():
            data.plot_key = PLOT_KEY_MAP[entry.cell_type][cond]
        categories[entry.cell_type].append(entry)

    tasks: list[Tuple[NeuronAnalysisResult, Dict, str, float, float]] = []
    for cell_type, items in categories.items():
        if "Other" in cell_type:
            continue
        for result in items:
            tasks.append((result, SPCConfig.to_dict(), str(args.output_dir), args.vmin, args.vmax))

    if tasks:
        if args.workers > 1:
            with mp.Pool(processes=args.workers) as pool:
                list(tqdm(pool.imap_unordered(visualize_neuron, tasks), total=len(tasks), desc="Rendering"))
        else:
            for task in tqdm(tasks, desc="Rendering"):
                visualize_neuron(task)

    elapsed = time.time() - start_time
    print(f"Saved visualisations to {args.output_dir} (elapsed {elapsed:.2f}s)")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
