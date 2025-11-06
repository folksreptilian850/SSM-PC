#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supplementary Video 5 Generator

Creates the two-part visualization for
"Spontaneous Formation of Grid-Cell-Like Representations".
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from datasets.navigation_dataset import SingleMazeDataset
from grid_cell_visualization import GridCellVisualizer, load_dataloader, load_model


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def decode_predictions(place_logits: torch.Tensor,
                       hd_logits: torch.Tensor,
                       place_cells,
                       hd_cells) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode logits into predicted positions and head directions."""
    place_probs = torch.softmax(place_logits, dim=-1)
    hd_probs = torch.softmax(hd_logits, dim=-1)

    cell_centers = place_cells.means.to(place_probs.device)
    pred_positions = torch.matmul(place_probs, cell_centers)

    hd_means = hd_cells.means.to(hd_probs.device)
    hd_complex = torch.exp(1j * hd_means)
    pred_hd_complex = torch.sum(hd_probs * hd_complex, dim=-1)
    pred_angles = torch.angle(pred_hd_complex)
    return pred_positions, pred_angles


def prepare_inference_input(positions: torch.Tensor,
                            angles: torch.Tensor,
                            velocities: torch.Tensor,
                            angular_vels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare model inputs for inference on a single trajectory."""
    velocity_input = torch.cat([velocities, angular_vels.unsqueeze(-1)], dim=-1).unsqueeze(0)
    init_pos = positions[0].unsqueeze(0)
    init_hd = angles[0].unsqueeze(0)
    return velocity_input, init_pos, init_hd


def resolve_font(preferred_family: str,
                 font_path: Optional[str] = None,
                 fallback_family: str = "Liberation Sans") -> str:
    """
    Resolve a usable font family for matplotlib.

    Returns the chosen family name and prints guidance when falling back.
    """
    if font_path:
        font_path = Path(font_path).expanduser().resolve()
        if not font_path.exists():
            raise FileNotFoundError(f"Specified font path does not exist: {font_path}")
        fm.fontManager.addfont(str(font_path))
        fm._rebuild()
        family = preferred_family or font_path.stem
        print(f"Loaded custom font from {font_path} as family '{family}'.")
        return family

    # First try current matplotlib cache
    try:
        fm.findfont(preferred_family, fallback_to_default=False)
        print(f"Using requested font family '{preferred_family}'.")
        return preferred_family
    except Exception:
        pass

    # Try discovering via fontconfig (fc-match)
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}\n", preferred_family],
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = result.stdout.strip()
        if candidate:
            font_path = Path(candidate)
            if font_path.exists():
                fm.fontManager.addfont(str(font_path))
                fm._rebuild()
                fm.findfont(preferred_family, fallback_to_default=False)
                print(f"Loaded '{preferred_family}' via fontconfig path {font_path}.")
                return preferred_family
    except Exception as fontconfig_err:
        print(f"⚠️  Unable to load '{preferred_family}' via fontconfig: {fontconfig_err}")

    print(
        f"⚠️  Font family '{preferred_family}' not found. "
        f"Falling back to '{fallback_family}'. "
        "Supply --font_path pointing to the desired TTF/OTF file to enforce usage."
    )
    fm.findfont(fallback_family, fallback_to_default=True)
    return fallback_family


class SupplementaryVideo5Generator:
    """Generates the requested two-part supplementary video."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir).resolve()
        self.frames_root = self.output_dir / "frames"
        self.heatmap_epoch_dir = self.frames_root / "heatmap_epochs"
        self.heatmap_frames_dir = self.frames_root / "segment1_frames"
        self.traj_frames_dir = self.frames_root / "segment2_frames"
        self.temp_dir = self.output_dir / "temp"
        self.video_part1 = self.output_dir / "video5_part1.mp4"
        self.video_part2 = self.output_dir / "video5_part2.mp4"
        self.final_video = self.output_dir / "video5_full.mp4"

        plt.style.use("default")
        resolved_family = resolve_font(
            preferred_family=args.font_family,
            font_path=args.font_path,
        )
        plt.rcParams["font.family"] = resolved_family
        plt.rcParams["font.sans-serif"] = [
            resolved_family,
            "Arial",
            "Liberation Sans",
            "DejaVu Sans",
        ]
        plt.rcParams["figure.dpi"] = 100
        self.font_family = resolved_family

        visualizer = GridCellVisualizer(
            save_dir=str(self.temp_dir / "grid_analysis"),
            nbins=args.nbins,
            env_size=args.env_size,
        )
        self.grid_scorer = visualizer.scorer
        self.autocorr_cmap = visualizer.autocorr_cmap
        # Match the Nature-style rate-map colors used in grid_viz_results-3 outputs
        self.rate_map_cmap = plt.get_cmap("jet")
        self.include_random_baseline = args.include_random_baseline
        self.random_baseline_epoch = args.random_baseline_epoch
        self.random_baseline_seed = args.random_baseline_seed

        self.checkpoint_dirs = [Path(p).resolve() for p in args.checkpoint_dirs]
        self.selected_neurons: List[int] = list(args.neurons)
        self.selected_trajectory_dir: Optional[str] = args.trajectory_dir
        self.cached_traj_results: Optional[List[Dict[str, np.ndarray]]] = None
        self.trajectory_sample: Optional[Dict[str, torch.Tensor]] = None
        self.gridness_tracking: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # General helpers
    # ------------------------------------------------------------------ #

    def reset_output_dirs(self) -> None:
        """Clean previous run artifacts inside the output directory."""
        for path in [
            self.frames_root,
            self.temp_dir,
            self.video_part1,
            self.video_part2,
            self.final_video,
        ]:
            if isinstance(path, Path):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
        ensure_dir(self.heatmap_epoch_dir)
        ensure_dir(self.heatmap_frames_dir)
        ensure_dir(self.traj_frames_dir)
        ensure_dir(self.temp_dir)

    def find_checkpoint(self, epoch: int) -> Path:
        """Locate the checkpoint file for a specific epoch."""
        candidates = [f"epoch_{epoch}.pth"]
        if epoch == -1:
            candidates.append("best_model.pth")
        for ckpt_dir in self.checkpoint_dirs:
            for name in candidates:
                candidate = ckpt_dir / name
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"Epoch {epoch} checkpoint not found in directories: "
            f"{[str(p) for p in self.checkpoint_dirs]}"
        )

    def encode_video(self, frames_dir: Path, output_path: Path, fps: int) -> None:
        """Encode frames into an mp4 clip using ffmpeg or OpenCV fallback."""
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            raise FileNotFoundError(
                f"No frames found in {frames_dir}. Cannot encode video."
            )

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            try:
                self._encode_with_ffmpeg(ffmpeg_path, frames_dir, output_path, fps)
                return
            except RuntimeError as err:
                print(f"⚠️  FFmpeg encoding failed: {err}\nFalling back to OpenCV.")
        else:
            print("⚠️  FFmpeg not available. Falling back to OpenCV video writer.")

        self._encode_with_opencv(frames, output_path, fps)

    def _encode_with_ffmpeg(self, ffmpeg_path: str, frames_dir: Path, output_path: Path, fps: int) -> None:
        """Encode using ffmpeg command line."""
        pattern = str(frames_dir / "frame_%04d.png")
        cmd = [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout.decode(errors='ignore')}\n"
                f"STDERR:\n{result.stderr.decode(errors='ignore')}"
            )

    def _encode_with_opencv(self, frames: Sequence[Path], output_path: Path, fps: int) -> None:
        """Encode frames using OpenCV VideoWriter."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for fallback encoding but is not installed. "
                "Install via 'pip install opencv-python' or ensure ffmpeg is available."
            ) from exc

        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            raise RuntimeError(f"Failed to load frame: {frames[0]}")
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter could not be opened.")

        try:
            for frame_path in frames:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"Failed to load frame: {frame_path}")
                writer.write(frame)
        finally:
            writer.release()

    def concat_videos(self, video_paths: Sequence[Path], output_path: Path) -> None:
        """Concatenate multiple mp4s (must share resolution/fps)."""
        concat_file = self.temp_dir / "concat.txt"
        with concat_file.open("w") as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # ------------------------------------------------------------------ #
    # Part 1 – Heatmap & SAC evolution
    # ------------------------------------------------------------------ #

    def collect_sequences(self) -> Dict[str, torch.Tensor]:
        """Collect a fixed subset of sequences for consistent per-epoch evaluation."""
        dataloader = load_dataloader(
            split=self.args.split,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        collected = {
            "positions": [],
            "angles": [],
            "velocities": [],
            "angular_velocities": [],
        }
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.args.num_batches:
                break
            for key in collected:
                collected[key].append(batch[key].clone())
        if not collected["positions"]:
            raise RuntimeError("Failed to collect any sequences for visualization.")
        return {k: torch.cat(v, dim=0) for k, v in collected.items()}

    def _collect_heatmap_results_with_model(
        self,
        model,
        sequences: Dict[str, torch.Tensor],
        neurons: Sequence[int],
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Collect rate maps and SACs for given neurons using a provided model."""
        if model is None:
            raise RuntimeError("Model instance is required to compute heatmap data.")
        model.eval()

        positions = sequences["positions"]
        angles = sequences["angles"]
        velocities = sequences["velocities"]
        angular_vels = sequences["angular_velocities"]

        pos_x_list: List[np.ndarray] = []
        pos_y_list: List[np.ndarray] = []
        neuron_values: Dict[int, List[np.ndarray]] = {idx: [] for idx in neurons}

        eval_batch_size = self.args.eval_batch_size
        total_samples = positions.shape[0]

        with torch.no_grad():
            for start in range(0, total_samples, eval_batch_size):
                end = min(start + eval_batch_size, total_samples)
                pos_batch = positions[start:end].to(self.device)
                ang_batch = angles[start:end].to(self.device)
                vel_batch = velocities[start:end].to(self.device)
                ang_vel_batch = angular_vels[start:end].to(self.device)

                vel_input = torch.cat([vel_batch, ang_vel_batch.unsqueeze(-1)], dim=-1)
                init_pos = pos_batch[:, 0]
                init_hd = ang_batch[:, 0]

                outputs = model(vel_input, init_pos, init_hd)
                bottleneck = outputs["bottleneck"].detach().cpu().numpy()

                pos_np = pos_batch.detach().cpu().numpy()
                pos_x_list.append(pos_np[:, :, 0])
                pos_y_list.append(pos_np[:, :, 1])

                for neuron_idx in neurons:
                    neuron_values[neuron_idx].append(bottleneck[:, :, neuron_idx])

        pos_x = np.concatenate([p.reshape(-1) for p in pos_x_list])
        pos_y = np.concatenate([p.reshape(-1) for p in pos_y_list])

        results: Dict[int, Dict[str, np.ndarray]] = {}
        for neuron_idx in neurons:
            activ = np.concatenate([a.reshape(-1) for a in neuron_values[neuron_idx]])
            ratemap = self.grid_scorer.calculate_ratemap(pos_x, pos_y, activ)
            sac = self.grid_scorer.calculate_sac(ratemap)
            score_60, score_90, mask_60, mask_90, _ = self.grid_scorer.get_scores(ratemap)
            results[neuron_idx] = {
                "ratemap": ratemap,
                "sac": sac,
                "gridness_60": score_60,
                "gridness_90": score_90,
                "mask_60": mask_60,
                "mask_90": mask_90,
            }

        return results

    def compute_random_baseline_data(
        self,
        sequences: Dict[str, torch.Tensor],
        neurons: Sequence[int],
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Compute visualization data for a randomly initialized model."""
        seed = self.random_baseline_seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        model = self.create_random_model()
        try:
            return self._collect_heatmap_results_with_model(model, sequences, neurons)
        finally:
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def compute_heatmap_data(
        self,
        sequences: Dict[str, torch.Tensor],
        epoch: int,
        neurons: Sequence[int],
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Compute rate maps, SACs, and gridness scores for selected neurons."""
        ckpt_path = self.find_checkpoint(epoch)
        model = load_model(str(ckpt_path), self.device)
        if model is None:
            raise RuntimeError(f"Unable to load model checkpoint for epoch {epoch} at {ckpt_path}.")
        try:
            return self._collect_heatmap_results_with_model(model, sequences, neurons)
        finally:
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def create_random_model(self) -> torch.nn.Module:
        """Create a randomly initialized GridCellNetwork instance."""
        from models.toroidal_grid_cell import GridCellNetwork
        from models.place_hd_cells import PlaceCellEnsemble, HeadDirectionCellEnsemble

        config = Config()
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
        ).to(self.device)
        model.eval()
        return model

    def compute_gridness_scores(
        self,
        sequences: Dict[str, torch.Tensor],
        epoch: int,
    ) -> np.ndarray:
        """Compute 60° gridness scores for all bottleneck neurons at a given epoch."""
        ckpt_path = self.find_checkpoint(epoch)
        model = load_model(str(ckpt_path), self.device)
        model.eval()

        positions = sequences["positions"]
        angles = sequences["angles"]
        velocities = sequences["velocities"]
        angular_vels = sequences["angular_velocities"]

        pos_list = []
        activation_list = []

        eval_batch_size = self.args.eval_batch_size
        total_samples = positions.shape[0]

        bottleneck_size: Optional[int] = None

        with torch.no_grad():
            for start in range(0, total_samples, eval_batch_size):
                end = min(start + eval_batch_size, total_samples)
                pos_batch = positions[start:end].to(self.device)
                ang_batch = angles[start:end].to(self.device)
                vel_batch = velocities[start:end].to(self.device)
                ang_vel_batch = angular_vels[start:end].to(self.device)

                vel_input = torch.cat([vel_batch, ang_vel_batch.unsqueeze(-1)], dim=-1)
                init_pos = pos_batch[:, 0]
                init_hd = ang_batch[:, 0]

                outputs = model(vel_input, init_pos, init_hd)
                bottleneck = outputs["bottleneck"].detach().cpu().numpy()  # [B, T, N]

                if bottleneck_size is None:
                    bottleneck_size = bottleneck.shape[-1]

                pos_np = pos_batch.detach().cpu().numpy()  # [B, T, 2]
                pos_list.append(pos_np.reshape(-1, 2))
                activation_list.append(bottleneck.reshape(-1, bottleneck.shape[-1]))

        if bottleneck_size is None:
            raise RuntimeError("Failed to compute bottleneck activations for gridness scoring.")

        all_positions = np.concatenate(pos_list, axis=0)
        all_activations = np.concatenate(activation_list, axis=0)

        scores = np.zeros(bottleneck_size, dtype=np.float32)
        for idx in range(bottleneck_size):
            ratemap = self.grid_scorer.calculate_ratemap(
                all_positions[:, 0],
                all_positions[:, 1],
                all_activations[:, idx],
            )
            score_60, _, _, _, _ = self.grid_scorer.get_scores(ratemap)
            scores[idx] = score_60

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return scores

    def auto_select_neurons(self, sequences: Dict[str, torch.Tensor]) -> List[int]:
        """Automatically pick neurons whose gridness grows from near-zero to high."""
        requested_epochs = self.args.gridness_selection_epochs
        available_epochs: List[int] = []
        for epoch in requested_epochs:
            try:
                self.find_checkpoint(epoch)
                available_epochs.append(epoch)
            except FileNotFoundError:
                print(f"⚠️  Skipping epoch {epoch} for neuron selection (checkpoint not found).")

        if len(available_epochs) < 2:
            raise ValueError(
                "Not enough checkpoints available for neuron auto-selection. "
                f"Found epochs: {available_epochs}"
            )

        print(f"Auto-selecting neurons using epochs: {available_epochs}")
        self.gridness_tracking = {}
        for epoch in available_epochs:
            scores = self.compute_gridness_scores(sequences, epoch)
            self.gridness_tracking[epoch] = scores
            print(f"Epoch {epoch}: gridness stats -> min {scores.min():.3f}, mean {scores.mean():.3f}, max {scores.max():.3f}")

        start_epoch = available_epochs[0]
        end_epoch = available_epochs[-1]
        start_scores = self.gridness_tracking[start_epoch]
        end_scores = self.gridness_tracking[end_epoch]

        low_thr = self.args.neuron_low_threshold
        high_thr = self.args.neuron_high_threshold
        high_cap = self.args.neuron_high_cap
        tolerance = self.args.neuron_monotonic_tolerance

        print(
            f"Filtering neurons with initial gridness ≤ {low_thr:.2f} and final gridness within "
            f"[{high_thr:.2f}, {high_cap:.2f}]"
        )

        metrics = []
        for idx in range(len(start_scores)):
            start_val = start_scores[idx]
            end_val = end_scores[idx]
            improvement = end_val - start_val

            monotonic = True
            prev = start_val
            for epoch in available_epochs[1:]:
                current = self.gridness_tracking[epoch][idx]
                if current + tolerance < prev:
                    monotonic = False
                    break
                prev = current

            metrics.append({
                "idx": idx,
                "start": start_val,
                "end": end_val,
                "improvement": improvement,
                "monotonic": monotonic,
            })

        def pick_candidates(filter_fn):
            candidates = [m for m in metrics if filter_fn(m)]
            candidates.sort(key=lambda m: m["improvement"], reverse=True)
            return [m["idx"] for m in candidates[: self.args.neuron_top_k]]

        candidate_indices = pick_candidates(
            lambda m: m["monotonic"] and m["start"] <= low_thr and high_thr <= m["end"] <= high_cap
        )
        if not candidate_indices:
            print("⚠️  Strict criteria yielded no neurons. Relaxing high cap constraint.")
            candidate_indices = pick_candidates(
                lambda m: m["monotonic"] and m["start"] <= low_thr and m["end"] >= high_thr
            )
        if not candidate_indices:
            print("⚠️  Relaxed criteria still empty. Falling back to top improvements (end ≥ threshold).")
            candidate_indices = pick_candidates(
                lambda m: m["end"] >= high_thr
            )
        if not candidate_indices:
            print("⚠️  Using absolute top improvements regardless of thresholds.")
            candidate_indices = pick_candidates(lambda m: True)

        print("Selected neurons (start score -> end score):")
        for idx in candidate_indices:
            start_val = start_scores[idx]
            end_val = end_scores[idx]
            print(f"  Neuron {idx}: {start_val:.3f} -> {end_val:.3f}")

        return candidate_indices

    def render_heatmap_figure(
        self,
        epoch: int,
        neuron_results: Dict[int, Dict[str, np.ndarray]],
        neurons: Sequence[int],
        save_path: Path,
        title_suffix: str = "",
    ) -> None:
        """Render the heatmap + SAC layout for a single epoch."""
        fig, axes = plt.subplots(
            2,
            len(neurons),
            figsize=(19.2, 10.8),
            constrained_layout=True,
        )
        base_title = f"Grid-Cell Emergence – Epoch {epoch}"
        title_text = f"{base_title} {title_suffix}".rstrip()
        fig.suptitle(title_text, fontsize=32, fontweight="bold")

        for col, neuron_idx in enumerate(neurons):
            data = neuron_results[neuron_idx]
            ratemap = data["ratemap"]
            sac = data["sac"]
            gridness = data["gridness_60"]
            mask = data["mask_60"]

            rm = np.nan_to_num(ratemap.copy())
            rm_min, rm_max = np.nanmin(rm), np.nanmax(rm)
            if rm_max > rm_min:
                rm = (rm - rm_min) / (rm_max - rm_min + 1e-9)

            ax_heat = axes[0, col]
            ax_heat.imshow(
                rm,
                origin="lower",
                extent=[0, self.args.env_size, 0, self.args.env_size],
                cmap=self.rate_map_cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax_heat.set_title(
                f"Neuron {neuron_idx}\nGridness (60°): {gridness:.2f}",
                fontsize=16,
            )
            ax_heat.set_xticks([])
            ax_heat.set_yticks([])
            if col == 0:
                ax_heat.set_ylabel("Spatial Firing Fields", fontsize=18)

            ax_sac = axes[1, col]
            useful_sac = sac * self.grid_scorer._plotting_sac_mask
            ax_sac.imshow(
                useful_sac,
                origin="lower",
                cmap=self.autocorr_cmap,
                vmin=-1.0,
                vmax=1.0,
                interpolation="nearest",
            )
            center = self.grid_scorer._nbins - 1
            for radius in mask:
                ax_sac.add_artist(
                    plt.Circle(
                        (center, center),
                        radius * self.grid_scorer._nbins,
                        fill=False,
                        color="white",
                        linewidth=1.2,
                        alpha=0.9,
                    )
                )
            ax_sac.set_xticks([])
            ax_sac.set_yticks([])
            if col == 0:
                ax_sac.set_ylabel("Spatial Autocorrelogram", fontsize=18)

        fig.savefig(str(save_path), dpi=100)
        plt.close(fig)

    def generate_heatmap_segment(self, sequences: Dict[str, torch.Tensor]) -> None:
        """Generate epoch panels and final frame sequence for the first segment."""
        neurons = self.selected_neurons
        if not neurons:
            raise RuntimeError("No neurons selected for heatmap visualization.")
        requested_epochs = list(self.args.heatmap_epochs)
        baseline_epoch = self.random_baseline_epoch

        epoch_order: List[int] = []
        seen_epochs = set()
        if self.include_random_baseline:
            epoch_order.append(baseline_epoch)
            seen_epochs.add(baseline_epoch)
        for epoch in requested_epochs:
            if epoch in seen_epochs:
                continue
            epoch_order.append(epoch)
            seen_epochs.add(epoch)

        if not epoch_order:
            raise RuntimeError("No epochs available for heatmap visualization.")

        epoch_to_frame: Dict[int, Path] = {}
        print("Rendering heatmap & SAC panels ...")
        for epoch in tqdm(epoch_order, desc="Heatmap Epochs"):
            if self.include_random_baseline and epoch == baseline_epoch:
                results = self.compute_random_baseline_data(sequences, neurons)
                suffix = "(Random Init)"
            else:
                results = self.compute_heatmap_data(sequences, epoch, neurons)
                suffix = ""

            save_path = self.heatmap_epoch_dir / f"epoch_{epoch:03d}.png"
            self.render_heatmap_figure(epoch, results, neurons, save_path, title_suffix=suffix)
            epoch_to_frame[epoch] = save_path

        total_frames = self.args.heatmap_frames
        if total_frames <= 0:
            return

        epoch_indices = np.linspace(0, len(epoch_order) - 1, total_frames)
        for frame_id, idx in enumerate(epoch_indices):
            order_idx = int(round(idx))
            order_idx = min(max(order_idx, 0), len(epoch_order) - 1)
            epoch = epoch_order[order_idx]
            src = epoch_to_frame[epoch]
            dst = self.heatmap_frames_dir / f"frame_{frame_id:04d}.png"
            shutil.copyfile(src, dst)

    # ------------------------------------------------------------------ #
    # Part 2 – Trajectory comparison
    # ------------------------------------------------------------------ #

    def load_single_trajectory(self, trajectory_dir: Path, sequence_length: int) -> Dict[str, torch.Tensor]:
        """Load one trajectory sample from the dataset."""
        dataset = SingleMazeDataset(
            str(trajectory_dir),
            sequence_length=sequence_length,
            stride=1,
            lazy_loading=True,
        )
        if len(dataset) == 0:
            raise ValueError(f"No valid sequences found in {trajectory_dir}")
        return dataset[0]

    def run_trajectory_inference(
        self,
        epoch: int,
        sample: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """Run trajectory inference for a given epoch."""
        ckpt_path = self.find_checkpoint(epoch)
        model = load_model(str(ckpt_path), self.device)
        model.eval()

        positions = sample["positions"]
        angles = sample["angles"]
        velocities = sample["velocities"]
        angular_vels = sample["angular_velocities"]

        max_steps = self.args.trajectory_steps
        steps = min(int(max_steps), positions.shape[0])
        if steps < 2:
            raise ValueError("Trajectory sample is too short for visualization.")

        positions = positions[:steps].clone()
        angles = angles[:steps].clone()
        velocities = velocities[:steps].clone()
        angular_vels = angular_vels[:steps].clone()

        velocity_input, init_pos, init_hd = prepare_inference_input(
            positions, angles, velocities, angular_vels
        )

        gt_positions = positions.cpu().numpy()
        gt_angles = angles.cpu().numpy()

        with torch.no_grad():
            outputs = model(
                velocity_input.to(self.device),
                init_pos.to(self.device),
                init_hd.to(self.device),
            )
            place_logits = outputs["place_logits"].squeeze(0)
            hd_logits = outputs["hd_logits"].squeeze(0)
            pred_positions, pred_angles = decode_predictions(
                place_logits,
                hd_logits,
                model.place_cells,
                model.hd_cells,
            )
            pred_positions = pred_positions.cpu().numpy()
            pred_angles = pred_angles.cpu().numpy()

        pred_positions = pred_positions[:steps]
        pred_angles = pred_angles[:steps]
        pred_positions[0] = gt_positions[0]
        pred_angles[0] = gt_angles[0]

        if gt_positions.shape[0] > 1:
            pos_errors = np.linalg.norm(gt_positions[1:] - pred_positions[1:], axis=1)
            mean_pos_error = float(np.mean(pos_errors))
            angle_errors = np.abs(np.angle(np.exp(1j * (gt_angles[1:] - pred_angles[1:]))))
            mean_angle_error = float(np.mean(angle_errors))
        else:
            mean_pos_error = 0.0
            mean_angle_error = 0.0

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "epoch": epoch,
            "gt_positions": gt_positions,
            "gt_angles": gt_angles,
            "pred_positions": pred_positions,
            "pred_angles": pred_angles,
            "mean_pos_error": mean_pos_error,
            "mean_angle_error": mean_angle_error,
        }

    def render_trajectory_frames(self, traj_data: Sequence[Dict[str, np.ndarray]]) -> None:
        """Render animated trajectory comparison frames."""
        env_size = self.args.env_size
        total_frames = self.args.trajectory_frames

        min_len = min(item["gt_positions"].shape[0] for item in traj_data)
        max_steps = min(self.args.trajectory_steps, min_len)
        if max_steps < 2:
            raise ValueError("Not enough steps in trajectory data to build visualization.")
        frame_indices = np.linspace(1, max_steps, total_frames).astype(int)

        for frame_id, step in enumerate(frame_indices):
            fig, axes = plt.subplots(
                1,
                len(traj_data),
                figsize=(19.2, 10.8),
                constrained_layout=True,
            )
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            fig.suptitle(
                f"Trajectory Comparison – Step {step}/{max_steps}",
                fontsize=30,
                fontweight="bold",
            )

            for ax, item in zip(axes, traj_data):
                gt = item["gt_positions"]
                pred = item["pred_positions"]
                epoch = item["epoch"]

                ax.plot(gt[:, 0], gt[:, 1], color="#1f77b4", alpha=0.2, linewidth=1.5, label="GT (full)")
                ax.plot(pred[:, 0], pred[:, 1], color="#ff7f0e", alpha=0.2, linewidth=1.5, label="Pred (full)")
                ax.plot(gt[:step, 0], gt[:step, 1], color="#1f77b4", linewidth=2.5, label="GT (up to t)")
                ax.plot(pred[:step, 0], pred[:step, 1], color="#ff7f0e", linewidth=2.5, label="Pred (up to t)")

                ax.scatter(gt[0, 0], gt[0, 1], color="#1f77b4", s=60, marker="o", zorder=5)
                ax.scatter(pred[step - 1, 0], pred[step - 1, 1], color="#ff7f0e", s=60, marker="^", zorder=5)

                ax.set_xlim(0, env_size)
                ax.set_ylim(0, env_size)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, linestyle="--", alpha=0.3)

                ax.set_title(
                    f"Epoch {epoch}\n"
                    f"Mean pos err: {item['mean_pos_error']:.2f} m\n"
                    f"Mean rot err: {item['mean_angle_error']:.2f} rad",
                    fontsize=16,
                )

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=14)

            save_path = self.traj_frames_dir / f"frame_{frame_id:04d}.png"
            fig.savefig(str(save_path), dpi=100)
            plt.close(fig)

    def generate_trajectory_segment(self) -> None:
        """Produce frames for the trajectory comparison segment."""
        if not self.selected_trajectory_dir:
            raise RuntimeError("Trajectory directory is not set.")

        config = Config()
        if self.trajectory_sample is None:
            self.trajectory_sample = self.load_single_trajectory(
                Path(self.selected_trajectory_dir),
                sequence_length=config.SEQUENCE_LENGTH,
            )

        if self.cached_traj_results is not None:
            traj_results = self.cached_traj_results
            print(f"Using cached trajectory metrics for {self.selected_trajectory_dir}.")
        else:
            traj_results = []
            print("Running trajectory inference across epochs ...")
            for epoch in tqdm(self.args.trajectory_epochs, desc="Trajectory Epochs"):
                result = self.run_trajectory_inference(epoch, self.trajectory_sample)
                traj_results.append(result)
            self.cached_traj_results = traj_results

        traj_results.sort(key=lambda x: x["epoch"])
        self.render_trajectory_frames(traj_results)

    # ------------------------------------------------------------------ #
    # Selection helpers
    # ------------------------------------------------------------------ #

    def determine_neurons(self, sequences: Dict[str, torch.Tensor]) -> List[int]:
        """Determine which neurons to visualize."""
        if self.args.auto_select_neurons:
            self.selected_neurons = self.auto_select_neurons(sequences)
        else:
            print("Auto neuron selection disabled. Using manually specified list.")
            self.selected_neurons = list(self.args.neurons)
            # Still record gridness progression for the manual selection
            self.gridness_tracking = {}
            requested_epochs = self.args.gridness_selection_epochs
            available_epochs: List[int] = []
            for epoch in requested_epochs:
                try:
                    self.find_checkpoint(epoch)
                    available_epochs.append(epoch)
                except FileNotFoundError:
                    print(f"⚠️  Skipping epoch {epoch} for manual gridness logging (checkpoint not found).")

            for epoch in available_epochs:
                scores = self.compute_gridness_scores(sequences, epoch)
                self.gridness_tracking[epoch] = scores
        print(f"Using neurons: {self.selected_neurons}")
        self.write_gridness_summary()
        return self.selected_neurons

    def write_gridness_summary(self) -> None:
        """Write gridness progression for selected neurons to CSV and console."""
        if not self.gridness_tracking or not self.selected_neurons:
            return

        ensure_dir(self.output_dir)
        csv_path = self.output_dir / "gridness_progress.csv"
        epochs = sorted(self.gridness_tracking.keys())

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "neuron", "gridness_60"])
            for epoch in epochs:
                scores = self.gridness_tracking[epoch]
                for neuron in self.selected_neurons:
                    writer.writerow([epoch, neuron, f"{scores[neuron]:.6f}"])

        # Console summary
        print("Gridness progression for selected neurons:")
        for neuron in self.selected_neurons:
            start_epoch = epochs[0]
            end_epoch = epochs[-1]
            start_score = self.gridness_tracking[start_epoch][neuron]
            end_score = self.gridness_tracking[end_epoch][neuron]
            print(f"  Neuron {neuron}: {start_score:.3f} -> {end_score:.3f}")
        print(f"Detailed gridness log written to: {csv_path}")

    def determine_trajectory(self) -> Tuple[str, Optional[Dict[str, torch.Tensor]], Optional[List[Dict[str, np.ndarray]]]]:
        """Determine trajectory directory and optionally cache metrics."""
        if self.args.auto_select_trajectory:
            path, sample, results = self.auto_select_trajectory_dir()
            self.selected_trajectory_dir = str(path)
            self.trajectory_sample = sample
            self.cached_traj_results = results
        else:
            if not self.selected_trajectory_dir:
                raise ValueError("trajectory_dir must be provided when auto selection is disabled.")
            path = Path(self.selected_trajectory_dir)
            if not path.exists():
                raise FileNotFoundError(f"Specified trajectory directory does not exist: {path}")
            sample, results = self.evaluate_trajectory_candidate(path)
            pos_errors = [r["mean_pos_error"] for r in results]
            rot_errors = [r["mean_angle_error"] for r in results]
            tol = self.args.trajectory_tolerance
            pos_monotonic = self._is_monotonic_decreasing(pos_errors, tol)
            rot_monotonic = self._is_monotonic_decreasing(rot_errors, tol)
            if not (pos_monotonic and rot_monotonic):
                print(
                    "⚠️  Provided trajectory does not exhibit strictly decreasing errors. "
                    "Consider enabling --auto_select_trajectory."
                )
            self.trajectory_sample = sample
            self.cached_traj_results = results
        print(f"Using trajectory directory: {self.selected_trajectory_dir}")
        return self.selected_trajectory_dir, self.trajectory_sample, self.cached_traj_results

    def auto_select_trajectory_dir(self) -> Tuple[Path, Dict[str, torch.Tensor], List[Dict[str, np.ndarray]]]:
        """Search the dataset for a trajectory with steadily improving errors."""
        config = Config()
        root = Path(config.DATA_ROOT)
        if not root.exists():
            raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")

        candidates: List[Path] = []
        for dataset_dir in sorted(root.iterdir()):
            if not dataset_dir.is_dir() or not dataset_dir.name.startswith("D"):
                continue
            for seq_dir in sorted(dataset_dir.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.isdigit():
                    candidates.append(seq_dir)
                    if len(candidates) >= self.args.trajectory_candidates:
                        break
            if len(candidates) >= self.args.trajectory_candidates:
                break

        if not candidates:
            raise RuntimeError("No trajectory directories found for auto-selection.")

        best_path: Optional[Path] = None
        best_score = -np.inf
        best_results: Optional[List[Dict[str, np.ndarray]]] = None
        best_sample: Optional[Dict[str, torch.Tensor]] = None

        tol = self.args.trajectory_tolerance

        print(f"Evaluating {len(candidates)} trajectory candidates for auto-selection ...")
        for seq_dir in candidates:
            try:
                sample, results = self.evaluate_trajectory_candidate(seq_dir)
            except Exception as exc:
                print(f"  Skipping {seq_dir}: {exc}")
                continue

            pos_errors = [r["mean_pos_error"] for r in results]
            rot_errors = [r["mean_angle_error"] for r in results]
            pos_monotonic = self._is_monotonic_decreasing(pos_errors, tol)
            rot_monotonic = self._is_monotonic_decreasing(rot_errors, tol)

            improvement_pos = pos_errors[0] - pos_errors[-1]
            improvement_rot = rot_errors[0] - rot_errors[-1]
            score = improvement_pos + 0.5 * improvement_rot

            print(
                f"  {seq_dir}: pos {pos_errors[0]:.3f}->{pos_errors[-1]:.3f}, "
                f"rot {rot_errors[0]:.3f}->{rot_errors[-1]:.3f}, "
                f"monotonic pos={pos_monotonic}, rot={rot_monotonic}"
            )

            if pos_monotonic and rot_monotonic and score > best_score:
                best_path = seq_dir
                best_score = score
                best_results = results
                best_sample = sample
            elif best_path is None and score > best_score:
                # Fall back to best score even if monotonicity not satisfied yet
                best_path = seq_dir
                best_score = score
                best_results = results
                best_sample = sample

        if best_path is None or best_results is None or best_sample is None:
            raise RuntimeError("Failed to identify a suitable trajectory for visualization.")

        print(f"Selected trajectory: {best_path} (score={best_score:.3f})")
        return best_path, best_sample, best_results

    def evaluate_trajectory_candidate(
        self,
        trajectory_dir: Path,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, np.ndarray]]]:
        """Evaluate trajectory errors across epochs for a given directory."""
        config = Config()
        sample = self.load_single_trajectory(trajectory_dir, sequence_length=config.SEQUENCE_LENGTH)

        results = []
        for epoch in self.args.trajectory_epochs:
            result = self.run_trajectory_inference(epoch, sample)
            results.append(result)
        return sample, results

    @staticmethod
    def _is_monotonic_decreasing(values: Sequence[float], tolerance: float) -> bool:
        """Check if a sequence is monotonically decreasing within tolerance."""
        prev = values[0]
        for val in values[1:]:
            if val > prev + tolerance:
                return False
            prev = val
        return True

    # ------------------------------------------------------------------ #
    # Pipeline entry
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Execute the end-to-end pipeline."""
        ensure_dir(self.output_dir)
        self.reset_output_dirs()

        print(f"Using device: {self.device}")
        print("Collecting sequences for heatmap analysis ...")
        sequences = self.collect_sequences()

        self.determine_neurons(sequences)

        self.generate_heatmap_segment(sequences)

        print("Encoding heatmap segment ...")
        self.encode_video(self.heatmap_frames_dir, self.video_part1, self.args.fps)

        if self.args.heatmap_only:
            shutil.copy2(self.video_part1, self.final_video)
            print("\n✅ Heatmap segment generated (trajectory segment skipped).")
            print(f"Part 1: {self.video_part1}")
            print(f"Final video: {self.final_video}")
            return

        self.determine_trajectory()
        self.generate_trajectory_segment()

        print("Encoding trajectory segment ...")
        self.encode_video(self.traj_frames_dir, self.video_part2, self.args.fps)

        print("Concatenating final video ...")
        self.concat_videos([self.video_part1, self.video_part2], self.final_video)

        print("\n✅ Supplementary Video 5 generated successfully!")
        print(f"Part 1: {self.video_part1}")
        print(f"Part 2: {self.video_part2}")
        print(f"Final video: {self.final_video}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Supplementary Video 5.")
    parser.add_argument(
        "--checkpoint_dirs",
        nargs="+",
        required=True,
        help="Checkpoint directories to search for epoch_N.pth files (searched in order).",
    )
    parser.add_argument(
        "--neurons",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[199, 113, 214, 161],
        help="Comma-separated neuron indices for heatmap visualization.",
    )
    parser.add_argument(
        "--heatmap_epochs",
        type=lambda s: [int(x) for x in s.split(",")],
        default=list(range(0, 51, 5)),
        help="Comma-separated epochs for the heatmap segment.",
    )
    parser.add_argument(
        "--trajectory_epochs",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[0, 5, 10, 20, 50],
        help="Comma-separated epochs for the trajectory segment.",
    )
    parser.add_argument(
        "--gridness_selection_epochs",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[0, 5, 10, 20, 30, 40, 50],
        help="Epochs used to judge neuron emergence when auto-selecting.",
    )
    parser.add_argument("--num_batches", type=int, default=50, help="Batches sampled for heatmap statistics.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for sequence collection.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size during model evaluation.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of dataloader workers.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--nbins", type=int, default=40, help="Rate-map resolution.")
    parser.add_argument("--env_size", type=float, default=15.0, help="Environment size (meters).")
    parser.add_argument("--fps", type=int, default=5, help="Frame rate for both video segments.")
    parser.add_argument("--heatmap_frames", type=int, default=100, help="Frame count for the heatmap segment.")
    parser.add_argument("--trajectory_frames", type=int, default=100, help="Frame count for the trajectory segment.")
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default=None,
        help="Directory containing the single trajectory data (e.g., .../D005178_P1/000000).",
    )
    parser.add_argument(
        "--trajectory_steps",
        type=int,
        default=50,
        help="Number of integration steps to visualize.",
    )
    parser.add_argument(
        "--trajectory_tolerance",
        type=float,
        default=0.02,
        help="Tolerance for treating errors as monotonically decreasing.",
    )
    parser.add_argument(
        "--trajectory_candidates",
        type=int,
        default=20,
        help="Maximum number of trajectory directories to evaluate when auto-selecting.",
    )
    parser.add_argument(
        "--auto_select_trajectory",
        action="store_true",
        help="Automatically search the dataset for a trajectory with improving accuracy.",
    )
    parser.add_argument(
        "--heatmap_only",
        action="store_true",
        help="Only generate the heatmap segment (skip trajectory visualization).",
    )
    parser.add_argument(
        "--disable_random_baseline",
        action="store_false",
        dest="include_random_baseline",
        help="Disable the prepended random-initialized epoch baseline in the heatmap segment.",
    )
    parser.add_argument(
        "--random_baseline_epoch",
        type=int,
        default=0,
        help="Epoch label to use for the random baseline visualization.",
    )
    parser.add_argument(
        "--random_baseline_seed",
        type=int,
        default=31415,
        help="Random seed used when constructing the random baseline model.",
    )
    parser.add_argument(
        "--disable_auto_neuron_selection",
        action="store_false",
        dest="auto_select_neurons",
        help="Disable automatic neuron emergence selection and use --neurons list as-is.",
    )
    parser.add_argument(
        "--neuron_low_threshold",
        type=float,
        default=0.1,
        help="Maximum initial gridness score when auto-selecting neurons.",
    )
    parser.add_argument(
        "--neuron_high_threshold",
        type=float,
        default=0.5,
        help="Minimum final gridness score when auto-selecting neurons.",
    )
    parser.add_argument(
        "--neuron_high_cap",
        type=float,
        default=0.9,
        help="Preferred maximum final gridness score when auto-selecting neurons.",
    )
    parser.add_argument(
        "--neuron_top_k",
        type=int,
        default=4,
        help="Number of neurons to visualize.",
    )
    parser.add_argument(
        "--neuron_monotonic_tolerance",
        type=float,
        default=0.05,
        help="Tolerance for monotonic gridness growth when auto-selecting neurons.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo-output/video5",
        help="Output directory for frames and videos.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device identifier.")
    parser.add_argument(
        "--font_family",
        type=str,
        default="Arial",
        help="Preferred font family for all text elements.",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default=None,
        help="Optional path to a specific TTF/OTF font file (e.g., Arial.ttf).",
    )
    parser.set_defaults(auto_select_neurons=True, include_random_baseline=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    generator = SupplementaryVideo5Generator(args)
    generator.run()


if __name__ == "__main__":
    main()
