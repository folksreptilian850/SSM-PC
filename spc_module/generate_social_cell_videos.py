import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from grid_cell.models.place_hd_cells import HeadDirectionCellEnsemble, PlaceCellEnsemble  # type: ignore
from spc_module.config import SPCConfig
from spc_module.social_grid_cell import SocialGridCellNetwork
from spc_module.trajectory import TrajectoryGenerator

matplotlib.use("Agg")

H264_CODECS = ("libx264", "libopenh264", "h264")
DEFAULT_NEURONS = [
    {"label": "Pure Place Cell", "neuron": 210, "type": "Pure Place Cell"},
    {"label": "Pure SPC", "neuron": 27, "type": "Pure SPC"},
    {"label": "Special SPC", "neuron": 145, "type": "Special SPC"},
]
TRAJECTORY_MAP = {
    "Pure SPC": {1: "self", 2: "peer", 3: "peer", 4: "peer"},
    "Special SPC": {1: "self", 2: "peer", 3: "self", 4: "peer"},
    "Pure Place Cell": {1: "self", 2: "peer", 3: "self", 4: "peer"},
}


def discover_ffmpeg_video_encoders() -> set[str]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-hide_banner", "-encoders"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Encoders:"):
            continue
        if stripped[0] != "V":
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            encoders.add(parts[1])
    return encoders


def resolve_codec(preferred_codec: str, available_encoders: set[str]) -> Tuple[str, str | None]:
    requested = preferred_codec.strip()
    if requested.lower() == "auto":
        for candidate in H264_CODECS:
            if candidate in available_encoders:
                return candidate, None
        raise RuntimeError("No H.264 encoder available (libx264/libopenh264). Install one or set --codec explicitly.")

    if requested in available_encoders:
        return requested, None

    if requested in H264_CODECS:
        for candidate in H264_CODECS:
            if candidate in available_encoders:
                note = f"Requested codec '{requested}' unavailable; falling back to '{candidate}'."
                return candidate, note
        raise RuntimeError(
            f"Requested codec '{requested}' unavailable and no alternative H.264 encoder found."
        )

    available_list = ", ".join(sorted(available_encoders)) or "unknown"
    raise RuntimeError(
        f"Requested codec '{requested}' is not supported by this FFmpeg build. Available encoders: {available_list}."
    )


def is_h264_codec(codec: str) -> bool:
    return codec in H264_CODECS


def cumulative_heatmaps(
    positions: np.ndarray,
    activations: np.ndarray,
    env_size: float,
    grid_size: int,
    sigma: float,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    num_steps = positions.shape[1]
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    counts = np.zeros_like(heatmap)
    frames = np.zeros((num_steps, grid_size, grid_size), dtype=np.float32)
    clipped = np.clip(activations, vmin, vmax)
    scale = (grid_size - 1) / env_size

    for step in range(num_steps):
        coords = positions[:, step, :]
        x_idx = np.clip((coords[:, 0] * scale).astype(int), 0, grid_size - 1)
        y_idx = np.clip((coords[:, 1] * scale).astype(int), 0, grid_size - 1)
        np.add.at(heatmap, (y_idx, x_idx), clipped[:, step])
        np.add.at(counts, (y_idx, x_idx), 1.0)
        averaged = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts > 0)
        frames[step] = gaussian_filter(averaged, sigma=sigma)

    return frames


def build_model(model_path: Path, device: torch.device) -> Tuple[SocialGridCellNetwork, SPCConfig]:
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


def sample_condition(
    model: SocialGridCellNetwork,
    generator: TrajectoryGenerator,
    device: torch.device,
    condition: int,
    num_reps: int,
) -> Dict[str, np.ndarray]:
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
    self_init_hd = torch.from_numpy(
        np.arctan2(samples["self_vel"][:, 0, 1], samples["self_vel"][:, 0, 0])
    ).float().to(device)
    peer_init_hd = torch.from_numpy(
        np.arctan2(samples["peer_vel"][:, 0, 1], samples["peer_vel"][:, 0, 0])
    ).float().to(device)

    with torch.no_grad():
        outputs = model(self_vel, self_init_pos, self_init_hd, peer_vel, peer_init_pos, peer_init_hd)

    return {
        "positions": samples,
        "activations": outputs["bottleneck_self"].cpu().numpy(),
    }


def layout_condition_figure(
    condition: int,
    config: SPCConfig,
    condition_data: Dict[str, np.ndarray],
    neuron_rows: List[Dict],
    heatmap_frames: Dict[Tuple[int, int], np.ndarray],
    output_path: Path,
    frame_rate: int,
    vmin: float,
    vmax: float,
    codec: str,
    bitrate: int,
    extra_args: List[str],
    dpi: int,
) -> None:
    env_size = config.ENV_SIZE
    seq_len = condition_data["positions"]["self_pos"].shape[1]

    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.3, 1.0, 1.0, 1.0], wspace=0.35)
    traj_ax = fig.add_subplot(gs[0, 0])
    heat_axes = [fig.add_subplot(gs[0, col]) for col in range(1, 4)]

    traj_ax.set_title(f"Condition {condition}", fontsize=12, pad=10)
    traj_ax.set_xlim(0, env_size)
    traj_ax.set_ylim(0, env_size)
    traj_ax.set_xticks(np.linspace(0, env_size, 6))
    traj_ax.set_yticks(np.linspace(0, env_size, 6))
    traj_ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)

    self_paths = condition_data["positions"]["self_pos"][:2]
    peer_paths = condition_data["positions"]["peer_pos"][:2]

    for idx, path in enumerate(self_paths):
        traj_ax.plot(path[:, 0], path[:, 1], color="cyan", linewidth=2.2, alpha=0.9 if idx == 0 else 0.6)
        traj_ax.scatter(path[0, 0], path[0, 1], color="cyan", edgecolors="black", s=65, marker="o", zorder=5)
    for idx, path in enumerate(peer_paths):
        traj_ax.plot(path[:, 0], path[:, 1], color="magenta", linewidth=2.2, alpha=0.9 if idx == 0 else 0.6)
        traj_ax.scatter(path[0, 0], path[0, 1], color="magenta", edgecolors="black", s=65, marker="^", zorder=5)

    legend_elements = [
        Line2D([0], [0], color="cyan", lw=2.2, label="Self path"),
        Line2D([0], [0], color="magenta", lw=2.2, label="Partner path"),
    ]
    traj_ax.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.85, fontsize=9)
    traj_ax.set_xlabel("X (m)")
    traj_ax.set_ylabel("Y (m)")

    self_lines = [
        traj_ax.plot([], [], color="cyan", linewidth=2.2, alpha=0.9 if idx == 0 else 0.6)[0]
        for idx in range(len(self_paths))
    ]
    peer_lines = [
        traj_ax.plot([], [], color="magenta", linewidth=2.2, alpha=0.9 if idx == 0 else 0.6)[0]
        for idx in range(len(peer_paths))
    ]
    self_markers = [
        traj_ax.scatter([], [], color="cyan", edgecolors="black", s=65, marker="o", zorder=5)
        for _ in range(len(self_paths))
    ]
    peer_markers = [
        traj_ax.scatter([], [], color="magenta", edgecolors="black", s=65, marker="^", zorder=5)
        for _ in range(len(peer_paths))
    ]

    heat_images = []
    for ax, neuron_info in zip(heat_axes, neuron_rows):
        frame_0 = heatmap_frames[(neuron_info["neuron"], condition)][0]
        img = ax.imshow(
            frame_0,
            origin="lower",
            extent=[0, env_size, 0, env_size],
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{neuron_info['label']}\nNeuron {neuron_info['neuron']}", fontsize=10)
        heat_images.append(img)

    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cbar = fig.colorbar(heat_images[0], cax=cbar_ax)
    cbar.set_label("Activation", rotation=270, labelpad=12)

    progress_text = fig.text(0.5, 0.02, f"Frame 1 / {seq_len}", ha="center", va="bottom", fontsize=10)

    def init():
        for line in self_lines + peer_lines:
            line.set_data([], [])
        for marker in self_markers + peer_markers:
            marker.set_offsets(np.array([[np.nan, np.nan]]))
        return [*self_lines, *peer_lines, *self_markers, *peer_markers, *heat_images]

    def update(frame_idx):
        for idx, line in enumerate(self_lines):
            path = self_paths[idx][: frame_idx + 1]
            line.set_data(path[:, 0], path[:, 1])
            self_markers[idx].set_offsets(path[-1])
        for idx, line in enumerate(peer_lines):
            path = peer_paths[idx][: frame_idx + 1]
            line.set_data(path[:, 0], path[:, 1])
            peer_markers[idx].set_offsets(path[-1])
        for img, neuron_info in zip(heat_images, neuron_rows):
            img.set_data(heatmap_frames[(neuron_info["neuron"], condition)][frame_idx])
        progress_text.set_text(f"Frame {frame_idx + 1} / {seq_len}")
        return [*self_lines, *peer_lines, *self_markers, *peer_markers, *heat_images]

    animation = FuncAnimation(fig, update, frames=seq_len, init_func=init, interval=1000 / frame_rate, blit=False)
    writer = FFMpegWriter(fps=frame_rate, metadata={"artist": "SSM-PC"}, codec=codec, bitrate=bitrate, extra_args=extra_args)
    animation.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate social cell demonstration videos.")
    parser.add_argument("--model_path", type=Path, required=True, help="Trained checkpoint path.")
    parser.add_argument("--output_dir", type=Path, default=Path("demo-output"), help="Directory for rendered videos.")
    parser.add_argument("--num_reps", type=int, default=60, help="Trajectories per condition.")
    parser.add_argument("--frame_rate", type=int, default=10, help="Video frame rate (FPS).")
    parser.add_argument("--grid_size", type=int, default=50, help="Heatmap resolution.")
    parser.add_argument("--sigma", type=float, default=1.3, help="Gaussian smoothing for heatmaps.")
    parser.add_argument("--vmin", type=float, default=0.0, help="Minimum activation for colour mapping.")
    parser.add_argument("--vmax", type=float, default=0.8, help="Maximum activation for colour mapping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory sampling.")
    parser.add_argument("--dpi", type=int, default=200, help="Video DPI (controls resolution).")
    parser.add_argument("--codec", type=str, default="auto", help="Preferred FFmpeg codec (auto selects H.264).")
    parser.add_argument("--bitrate", type=int, default=4000, help="Bitrate in kbps.")
    parser.add_argument("--pix_fmt", type=str, default="yuv420p", help="FFmpeg pixel format.")
    parser.add_argument("--ffmpeg_profile", type=str, default="high", help="libx264 profile (baseline/main/high).")
    parser.add_argument("--ffmpeg_preset", type=str, default="", help="libx264 preset (optional).")
    args = parser.parse_args()

    available_encoders = discover_ffmpeg_video_encoders()
    if not available_encoders:
        raise RuntimeError("Unable to query FFmpeg encoders. Ensure ffmpeg is installed and on PATH.")

    codec, codec_note = resolve_codec(args.codec, available_encoders)
    if codec_note:
        print(codec_note)
    h264_selected = is_h264_codec(codec)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = build_model(args.model_path, device)
    generator = TrajectoryGenerator(config)

    neuron_rows = DEFAULT_NEURONS

    heatmap_frames: Dict[Tuple[int, int], np.ndarray] = {}
    condition_store: Dict[int, Dict[str, np.ndarray]] = {}

    for condition in [1, 2, 3, 4]:
        condition_data = sample_condition(model, generator, device, condition, args.num_reps)
        condition_store[condition] = condition_data
        seq_len = condition_data["positions"]["self_pos"].shape[1]

        for row in neuron_rows:
            key = TRAJECTORY_MAP[row["type"]][condition]
            positions = condition_data["positions"][f"{key}_pos"]
            activations = condition_data["activations"][:, :, row["neuron"]]
            frames = cumulative_heatmaps(
                positions,
                activations,
                env_size=config.ENV_SIZE,
                grid_size=args.grid_size,
                sigma=args.sigma,
                vmin=args.vmin,
                vmax=args.vmax,
            )
            if frames.shape[0] != seq_len:
                raise RuntimeError("Unexpected heatmap frame count")
            heatmap_frames[(row["neuron"], condition)] = frames

    extra_args = ["-pix_fmt", args.pix_fmt, "-movflags", "+faststart"]
    if args.ffmpeg_profile and h264_selected:
        extra_args.extend(["-profile:v", args.ffmpeg_profile])
    if args.ffmpeg_preset and codec == "libx264":
        extra_args.extend(["-preset", args.ffmpeg_preset])

    for condition in [1, 2, 3, 4]:
        output_path = args.output_dir / f"condition_{condition}_progress.mp4"
        layout_condition_figure(
            condition,
            config,
            condition_store[condition],
            neuron_rows,
            heatmap_frames,
            output_path,
            args.frame_rate,
            args.vmin,
            args.vmax,
            codec,
            args.bitrate,
            extra_args,
            args.dpi,
        )


if __name__ == "__main__":
    main()
