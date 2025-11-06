import csv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import correlate2d
from sklearn.neighbors import KernelDensity
from matplotlib.colors import LinearSegmentedColormap

class NavigationVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        


        cdict = {
            'red': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
            'green': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
            'blue': [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]
        }
        self.autocorr_cmap = LinearSegmentedColormap('BlueWhiteRed', cdict)
    
    def compute_spatial_autocorr(self, activation_map):









        activation_map = activation_map - np.mean(activation_map)

        autocorr = correlate2d(activation_map, activation_map, mode='same', boundary='wrap')
        

        center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2
        center_val = autocorr[center_y, center_x]
        if abs(center_val) > 1e-10:
            autocorr = autocorr / center_val
        return autocorr

    def compute_gridness(self, autocorr):












        center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2
        

        max_radius = min(center_y, center_x) // 2
        min_radius = max(3, max_radius // 4)
        
        y, x = np.ogrid[-center_y:autocorr.shape[0]-center_y, -center_x:autocorr.shape[1]-center_x]
        radius = np.sqrt(x*x + y*y)
        ring_mask = (radius >= min_radius) & (radius <= max_radius)
        

        ring_area = autocorr.copy()
        ring_area[~ring_mask] = 0
        

        angles_pos = [60, 120]
        angles_neg = [30, 90, 150]
        
        corrs_pos = []
        corrs_neg = []
        
        for angle in angles_pos:
            rotated = rotate(ring_area, angle, reshape=False, order=3)

            valid_mask = ring_mask & (rotated != 0)
            if np.sum(valid_mask) > 0:
                orig_valid = ring_area[valid_mask].flatten()
                rot_valid = rotated[valid_mask].flatten()
                if len(orig_valid) > 1:
                    corr = np.corrcoef(orig_valid, rot_valid)[0, 1]
                    corrs_pos.append(corr)
        
        for angle in angles_neg:
            rotated = rotate(ring_area, angle, reshape=False, order=3)
            valid_mask = ring_mask & (rotated != 0)
            if np.sum(valid_mask) > 0:
                orig_valid = ring_area[valid_mask].flatten()
                rot_valid = rotated[valid_mask].flatten()
                if len(orig_valid) > 1:
                    corr = np.corrcoef(orig_valid, rot_valid)[0, 1]
                    corrs_neg.append(corr)
        

        if len(corrs_pos) == 0 or len(corrs_neg) == 0:
            return -1.0
            

        gridness = np.mean(corrs_pos) - np.mean(corrs_neg)
        return gridness

    def compute_ratemap_kde(self, positions, activations, env_size=15, resolution=50, bandwidth=0.5):














        x = np.linspace(0, env_size, resolution)
        y = np.linspace(0, env_size, resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.vstack([Y.ravel(), X.ravel()]).T  # [resolution^2, 2]
        

        positions_norm = positions / env_size
        

        weight_map = np.zeros((resolution, resolution))
        count_map = np.zeros((resolution, resolution))
        

        for pos, act in zip(positions, activations):

            ix = min(int(pos[0] / env_size * (resolution-1)), resolution-1)
            iy = min(int(pos[1] / env_size * (resolution-1)), resolution-1)
            
            weight_map[iy, ix] += act
            count_map[iy, ix] += 1
        

        mask = count_map > 0
        ratemap = np.zeros((resolution, resolution))
        ratemap[mask] = weight_map[mask] / count_map[mask]
        

        smoothed_ratemap = gaussian_filter(ratemap, sigma=bandwidth)
        
        return smoothed_ratemap

    def collect_trajectories_and_activations(self, model, dataloader, device, num_batches=10):














        model.eval()
        positions_list = []
        bottleneck_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                positions = batch['positions'].to(device)
                angles = batch['angles'].to(device)
                velocities = batch['velocities'].to(device)
                ang_vels = batch['angular_velocities'].to(device)
                

                w = ang_vels.unsqueeze(-1)
                velocity_input = torch.cat([velocities, w], dim=-1)
                init_pos = positions[:, 0]
                init_hd = angles[:, 0]
                

                outputs = model(velocity_input, init_pos, init_hd)
                

                for b in range(positions.size(0)):

                    pos_sample = positions[b, 1:].cpu().numpy()
                    bottleneck_sample = outputs['bottleneck'][b, 1:].cpu().numpy()
                    
                    positions_list.append(pos_sample)
                    bottleneck_list.append(bottleneck_sample)
        
        return positions_list, bottleneck_list

    def visualize_grid_cells(self, model, dataloader, device, epoch, num_batches=10, num_neurons=6, suffix=""):
















        positions_list, bottleneck_list = self.collect_trajectories_and_activations(
            model, dataloader, device, num_batches)
        

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            bottleneck_size = model.module.bottleneck_size
        else:
            bottleneck_size = model.bottleneck_size
        

        gridness_scores = []
        all_positions = np.concatenate([p.reshape(-1, 2) for p in positions_list], axis=0)
        

        all_bottleneck = np.concatenate([b.reshape(-1, bottleneck_size) for b in bottleneck_list], axis=0)
        

        candidate_neurons = np.random.choice(bottleneck_size, min(bottleneck_size, 30), replace=False)
        candidate_gridness = []
        
        resolution = 50
        
        for neuron_idx in candidate_neurons:
            activations = all_bottleneck[:, neuron_idx]
            

            ratemap = self.compute_ratemap_kde(all_positions, activations, resolution=resolution)
            

            autocorr = self.compute_spatial_autocorr(ratemap)
            

            gridness = self.compute_gridness(autocorr)
            candidate_gridness.append((neuron_idx, gridness))
        

        candidate_gridness.sort(key=lambda x: x[1], reverse=True)
        

        selected_neurons = [n for n, _ in candidate_gridness[:num_neurons]]
        

        fig = plt.figure(figsize=(10, num_neurons * 2.5))
        gridspec = fig.add_gridspec(num_neurons, 3, width_ratios=[1, 1, 0.05])
        
        for i, neuron_idx in enumerate(selected_neurons):
            activations = all_bottleneck[:, neuron_idx]
            

            ratemap = self.compute_ratemap_kde(all_positions, activations, resolution=resolution)
            

            autocorr = self.compute_spatial_autocorr(ratemap)
            

            gridness = self.compute_gridness(autocorr)
            gridness_scores.append(gridness)
            

            ax1 = fig.add_subplot(gridspec[i, 0])
            im1 = ax1.imshow(ratemap, origin='lower', extent=[0, 15, 0, 15], 
                        cmap='viridis', interpolation='bilinear')
            ax1.set_title(f"Neuron {neuron_idx} Ratemap (t=1:)")
            ax1.set_xlabel("X position (m)")
            if i == 0:
                ax1.set_ylabel("Y position (m)")
            

            ax2 = fig.add_subplot(gridspec[i, 1])
            im2 = ax2.imshow(autocorr, origin='lower', extent=[-15, 15, -15, 15], 
                        cmap=self.autocorr_cmap, vmin=-1, vmax=1, interpolation='bilinear')
            ax2.set_title(f"Autocorrelation (Gridness: {gridness:.2f})")
            ax2.set_xlabel("X lag (m)")
            if i == 0:
                ax2.set_ylabel("Y lag (m)")
            

            if i == num_neurons - 1:
                cax = fig.add_subplot(gridspec[:, 2])
                plt.colorbar(im2, cax=cax, label="Correlation")
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'grid_cells_epoch_{epoch}_{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        

        avg_gridness = np.mean(gridness_scores) if gridness_scores else 0.0
        print(f"Epoch {epoch}-{suffix}: Average gridness score of top {num_neurons} neurons: {avg_gridness:.3f}")
        

        self.visualize_hexagonal_patterns(selected_neurons, all_positions, all_bottleneck, epoch)
        
        return avg_gridness

    def visualize_hexagonal_patterns(self, selected_neurons, all_positions, all_bottleneck, epoch):











        import matplotlib.pyplot as plt
        from matplotlib.patches import RegularPolygon


        max_plot = min(len(selected_neurons), 6)


        fig = plt.figure(figsize=(8, max_plot * 3))
        gs = fig.add_gridspec(max_plot, 2, wspace=0.3, hspace=0.4)

        resolution = 50
        env_size = 15

        for i, neuron_idx in enumerate(selected_neurons[:max_plot]):

            activations = all_bottleneck[:, neuron_idx]


            ratemap = self.compute_ratemap_kde(all_positions, activations, resolution=resolution)
            autocorr = self.compute_spatial_autocorr(ratemap)
            gridness = self.compute_gridness(autocorr)


            ax_left = fig.add_subplot(gs[i, 0])
            extent_val = [-env_size, env_size, -env_size, env_size]
            im_left = ax_left.imshow(autocorr, origin='lower',
                                    extent=extent_val,
                                    cmap=self.autocorr_cmap, vmin=-1, vmax=1,
                                    interpolation='bilinear')
            ax_left.set_title(f"Neuron {neuron_idx} (t=1:)\nAutocorr (Gridness: {gridness:.2f})")
            ax_left.set_xlabel("X lag (m)")
            ax_left.set_ylabel("Y lag (m)")


            ax_right = fig.add_subplot(gs[i, 1])
            im_right = ax_right.imshow(autocorr, origin='lower',
                                    extent=extent_val,
                                    cmap=self.autocorr_cmap, vmin=-1, vmax=1,
                                    interpolation='bilinear')
            ax_right.set_title("Hex Overlay")
            ax_right.set_xlabel("X lag (m)")
            ax_right.set_ylabel("Y lag (m)")


            center_x = 0.0
            center_y = 0.0
            max_radius = env_size

            for angle_deg in [0, 60, 120, 180, 240, 300]:
                angle_rad = np.deg2rad(angle_deg)
                dx = max_radius * np.cos(angle_rad)
                dy = max_radius * np.sin(angle_rad)
                ax_right.plot([center_x, center_x + dx],
                            [center_y, center_y + dy],
                            color='black', linestyle='--', linewidth=1.0)


            hex_radius = env_size * 0.5
            hex_patch = RegularPolygon((center_x, center_y),
                                    numVertices=6,
                                    radius=hex_radius,
                                    orientation=-np.pi/2,
                                    fill=False,
                                    edgecolor='blue',
                                    linewidth=1.2)
            ax_right.add_patch(hex_patch)


            if i == max_plot - 1:
                cbar = plt.colorbar(im_right, ax=[ax_left, ax_right], fraction=0.045)
                cbar.set_label("Correlation")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'hex_patterns_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Hexagonal pattern visualization saved at: {save_path}")


    def visualize_epoch(self, epoch, model, dataloader, device, metrics_history, suffix=""):














        self.plot_metrics(metrics_history, epoch, suffix)
        

        avg_gridness = self.visualize_grid_cells(model, dataloader, device, epoch, num_batches=10, suffix=suffix)
        
        return avg_gridness

    def plot_metrics(self, metrics_history, epoch, suffix=""):
        """Create a Nature-style convergence figure for loss metrics."""
        required_keys = {
            "Total Loss": ("train_total", "val_total"),
            "Place Loss": ("train_place", "val_place"),
            "Head Direction Loss": ("train_hd", "val_hd"),
        }

        missing = [
            key
            for title_keys in required_keys.values()
            for key in title_keys
            if key not in metrics_history or len(metrics_history[key]) == 0
        ]
        if missing:
            raise ValueError(
                f"plot_metrics missing required history entries: {sorted(set(missing))}"
            )

        epochs = np.arange(1, len(metrics_history["train_total"]) + 1)
        nature_rc = {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "legend.frameon": False,
        }

        train_color = "#1f77b4"  # deep blue
        val_color = "#ff7f0e"    # orange

        panel_records = []

        with plt.rc_context(nature_rc):
            fig, axes = plt.subplots(
                1,
                3,
                sharex=True,
                figsize=(12.5, 3.8),
                constrained_layout=True,
            )

            for ax, (title, (train_key, val_key)) in zip(axes, required_keys.items()):
                train_values = np.asarray(metrics_history[train_key], dtype=float)
                val_values = np.asarray(metrics_history[val_key], dtype=float)

                ax.plot(
                    epochs,
                    train_values,
                    color=train_color,
                    linewidth=2.2,
                    label="Train Loss",
                )
                ax.plot(
                    epochs,
                    val_values,
                    color=val_color,
                    linewidth=2.2,
                    linestyle="--",
                    label="Val Loss",
                )

                ax.set_title(title, fontweight="bold")
                ax.set_xlabel("Epoch")
                ax.set_xlim(epochs[0], epochs[-1])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

                # Highlight the converged values at the final epoch.
                ax.scatter(epochs[-1], train_values[-1], color=train_color, s=28, zorder=3)
                ax.scatter(epochs[-1], val_values[-1], color=val_color, s=28, zorder=3)

                final_annotation = {
                    "xytext": (5, 12),
                    "textcoords": "offset points",
                    "fontsize": 9,
                    "color": "#333333",
                }
                ax.annotate(
                    f"{train_values[-1]:.3f}",
                    xy=(epochs[-1], train_values[-1]),
                    ha="left",
                    va="bottom",
                    **final_annotation,
                )
                ax.annotate(
                    f"{val_values[-1]:.3f}",
                    xy=(epochs[-1], val_values[-1]),
                    ha="left",
                    va="top",
                    **final_annotation,
                )

                if ax is axes[0]:
                    ax.set_ylabel("Loss")
                    ax.text(
                        -0.18,
                        1.05,
                        "Panel A",
                        transform=ax.transAxes,
                        fontsize=13,
                        fontweight="bold",
                        va="top",
                    )

                panel_records.append((title, train_values, val_values))

            axes[1].legend(loc="upper right", handlelength=2.8)
            fig.suptitle(
                "Figure S1-A: Model Convergence on the Path Integration Task",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )

            suffix_clean = f"_{suffix}" if suffix else ""
            basename = f"figure_s1a_convergence_epoch_{epoch}{suffix_clean}"
            png_path = os.path.join(self.save_dir, f"{basename}.png")
            svg_path = os.path.join(self.save_dir, f"{basename}.svg")

            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(svg_path, format="svg", bbox_inches="tight")

        plt.close(fig)

        csv_map = {
            "Total Loss": "total_loss",
            "Place Loss": "place_loss",
            "Head Direction Loss": "head_direction_loss",
        }

        for title, train_values, val_values in panel_records:
            csv_name = csv_map.get(title, title.lower().replace(" ", "_"))
            csv_path = os.path.join(self.save_dir, f"{basename}_{csv_name}.csv")
            with open(csv_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["epoch", "train_loss", "val_loss"])
                for epoch_idx, train_val, val_val in zip(epochs, train_values, val_values):
                    writer.writerow([
                        int(epoch_idx),
                        f"{train_val:.6f}",
                        f"{val_val:.6f}",
                    ])
