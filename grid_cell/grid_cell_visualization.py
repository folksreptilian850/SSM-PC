#!/usr/bin/env python
# -*- coding: utf-8 -*-






import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import rotate
import scipy.signal
import scipy.stats


plt.style.use('default')
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12


def circle_mask(size, radius, in_val=1.0, out_val=0.0):

    sz = [int(size[0] / 2), int(size[1] / 2)]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0)
    x = x.repeat(size[0], 0)
    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1)
    y = y.repeat(size[1], 1)
    z = np.sqrt(x**2 + y**2)
    z = np.less_equal(z, radius)
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(z)


class GridScorer:


    def __init__(self, nbins, coords_range, mask_parameters, min_max=False):








        self._nbins = nbins
        self._min_max = min_max
        self._coords_range = coords_range
        self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
        

        self._masks = [(self._get_ring_mask(mask_min, mask_max), (mask_min, mask_max))
                      for mask_min, mask_max in mask_parameters]
        

        self._plotting_sac_mask = circle_mask(
            [self._nbins * 2 - 1, self._nbins * 2 - 1],
            self._nbins,
            in_val=1.0,
            out_val=np.nan)

    def calculate_ratemap(self, xs, ys, activations, statistic='mean'):

        return scipy.stats.binned_statistic_2d(
            xs,
            ys,
            activations,
            bins=self._nbins,
            statistic=statistic,
            range=self._coords_range)[0]

    def _get_ring_mask(self, mask_min, mask_max):

        n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
        return (circle_mask(n_points, mask_max * self._nbins) *
                (1 - circle_mask(n_points, mask_min * self._nbins)))

    def grid_score_60(self, corr):

        if self._min_max:
            return np.minimum(corr[60], corr[120]) - np.maximum(
                corr[30], np.maximum(corr[90], corr[150]))
        else:
            return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

    def grid_score_90(self, corr):

        return corr[90] - (corr[45] + corr[135]) / 2

    def calculate_sac(self, seq1):

        seq2 = seq1

        def filter2(b, x):
            stencil = np.rot90(b, 2)
            return scipy.signal.convolve2d(x, stencil, mode='full')

        seq1 = np.nan_to_num(seq1)
        seq2 = np.nan_to_num(seq2)

        ones_seq1 = np.ones(seq1.shape)
        ones_seq1[np.isnan(seq1)] = 0
        ones_seq2 = np.ones(seq2.shape)
        ones_seq2[np.isnan(seq2)] = 0

        seq1[np.isnan(seq1)] = 0
        seq2[np.isnan(seq2)] = 0

        seq1_sq = np.square(seq1)
        seq2_sq = np.square(seq2)

        seq1_x_seq2 = filter2(seq1, seq2)
        sum_seq1 = filter2(seq1, ones_seq2)
        sum_seq2 = filter2(ones_seq1, seq2)
        sum_seq1_sq = filter2(seq1_sq, ones_seq2)
        sum_seq2_sq = filter2(ones_seq1, seq2_sq)
        n_bins = filter2(ones_seq1, ones_seq2)
        n_bins_sq = np.square(n_bins)

        std_seq1 = np.power(
            np.subtract(
                np.divide(sum_seq1_sq, n_bins),
                (np.divide(np.square(sum_seq1), n_bins_sq))), 0.5)
        std_seq2 = np.power(
            np.subtract(
                np.divide(sum_seq2_sq, n_bins),
                (np.divide(np.square(sum_seq2), n_bins_sq))), 0.5)
        covar = np.subtract(
            np.divide(seq1_x_seq2, n_bins),
            np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
        x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
        x_coef = np.real(x_coef)
        x_coef = np.nan_to_num(x_coef)
        return x_coef

    def rotated_sacs(self, sac, angles):

        return [
            scipy.ndimage.rotate(sac, angle, reshape=False)
            for angle in angles
        ]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):

        masked_sac = sac * mask
        ring_area = np.sum(mask)

        masked_sac_mean = np.sum(masked_sac) / ring_area

        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
        corrs = dict()
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
        return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

    def get_scores(self, rate_map):

        sac = self.calculate_sac(rate_map)
        rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

        scores = [
            self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
            for mask, mask_params in self._masks
        ]
        scores_60, scores_90, variances = map(np.asarray, zip(*scores))
        max_60_ind = np.argmax(scores_60)
        max_90_ind = np.argmax(scores_90)

        return (scores_60[max_60_ind], scores_90[max_90_ind],
                self._masks[max_60_ind][1], self._masks[max_90_ind][1], sac)

    def plot_ratemap(self, ratemap, ax=None, title=None, **kwargs):

        if ax is None:
            ax = plt.gca()
        ax.imshow(ratemap, interpolation='none', **kwargs)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)

    def plot_sac(self, sac, mask_params=None, ax=None, title=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        useful_sac = sac * self._plotting_sac_mask
        ax.imshow(useful_sac, interpolation='none', **kwargs)
        

        if mask_params is not None:
            center = self._nbins - 1
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[0] * self._nbins,
                    fill=False,
                    edgecolor='k'))
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[1] * self._nbins,
                    fill=False,
                    edgecolor='k'))
        ax.axis('off')
        if title is not None:
            ax.set_title(title)


class GridCellVisualizer:
    def __init__(self, save_dir, nbins=40, env_size=15):







        self.save_dir = save_dir
        self.nbins = nbins
        self.env_size = env_size
        os.makedirs(save_dir, exist_ok=True)
        

        self.coords_range = ((0, env_size), (0, env_size))
        

        starts = [0.1] * 15
        ends = np.linspace(0.2, 0.85, num=15)
        mask_parameters = list(zip(starts, ends.tolist()))
        

        self.scorer = GridScorer(nbins, self.coords_range, mask_parameters)
        


        cdict = {
            'red':   [(0.0, 0.0, 0.0),
                      (0.3, 0.0, 0.0),
                      (0.5, 0.5, 0.5),
                      (0.7, 1.0, 1.0),
                      (1.0, 0.8, 0.8)],
                      
            'green': [(0.0, 0.0, 0.0),
                      (0.3, 0.8, 0.8),
                      (0.5, 0.8, 0.8),
                      (0.7, 1.0, 1.0),
                      (1.0, 0.0, 0.0)],
                      
            'blue':  [(0.0, 0.8, 0.8),
                      (0.3, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (0.7, 0.0, 0.0),
                      (1.0, 0.0, 0.0)]
        }
        self.autocorr_cmap = LinearSegmentedColormap('GridCellMap', cdict)
    
    def collect_grid_cell_data(self, model, dataloader, device, num_batches=20):












        print(f"收集{num_batches}个批次的数据用于可视化...")
        
        model.eval()
        positions_list = []
        bottleneck_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="处理批次")):
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

                    pos_sample = positions[b].cpu().numpy()  # [seq_len, 2]
                    bottleneck_sample = outputs['bottleneck'][b].cpu().numpy()  # [seq_len, bottleneck_size]
                    
                    positions_list.append(pos_sample)
                    bottleneck_list.append(bottleneck_sample)
        
        print(f"收集完成！总共收集了{len(positions_list)}个样本序列")
        return positions_list, bottleneck_list
    
    def get_scores_and_plot(self, positions, activations, filename):














        n_units = activations.shape[1]
        

        ratemaps = []
        for i in tqdm(range(n_units), desc="计算Rate Maps"):
            ratemap = self.scorer.calculate_ratemap(
                positions[:, 0], positions[:, 1], activations[:, i])
            ratemaps.append(ratemap)
        

        results = [self.scorer.get_scores(rate_map) for rate_map in tqdm(ratemaps, desc="计算Grid Scores")]
        scores_60, scores_90, max_60_mask, max_90_mask, sacs = zip(*results)
        

        ordering = np.argsort(-np.array(scores_60))
        

        cols = 8
        neurons_per_page = 16
        total_pages = int(np.ceil(n_units / neurons_per_page))
        

        pdf_path = os.path.join(self.save_dir, filename)
        with PdfPages(pdf_path) as pdf:
            for page in range(total_pages):
                start_idx = page * neurons_per_page
                end_idx = min(start_idx + neurons_per_page, n_units)
                page_ordering = ordering[start_idx:end_idx]
                
                n_on_page = len(page_ordering)
                rows = int(np.ceil(n_on_page / cols)) * 2
                
                fig = plt.figure(figsize=(cols * 3, rows * 3))
                
                for i, idx in enumerate(page_ordering):

                    ax_rate = plt.subplot(rows, cols, i + 1)
                    title = f"Neuron {idx} (60°: {scores_60[idx]:.2f})"
                    self.scorer.plot_ratemap(ratemaps[idx], ax=ax_rate, title=title, cmap='viridis')
                    

                    ax_sac = plt.subplot(rows, cols, i + 1 + cols)
                    self.scorer.plot_sac(
                        sacs[idx],
                        mask_params=max_60_mask[idx],
                        ax=ax_sac,
                        title=f"Mask: {max_60_mask[idx]}",
                        cmap=self.autocorr_cmap,
                        vmin=-1, vmax=1
                    )
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            

            ax1.hist(scores_60, bins=30)
            ax1.set_title('60° Gridness Score Distribution')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Count')
            ax1.axvline(x=0, color='r', linestyle='--')
            

            ax2.hist(scores_90, bins=30)
            ax2.set_title('90° Gridness Score Distribution')
            ax2.set_xlabel('Score')
            ax2.axvline(x=0, color='r', linestyle='--')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        print(f"可视化已保存到: {pdf_path}")
        

        return (np.array(scores_60), np.array(scores_90),
                np.array([np.mean(m) for m in max_60_mask]),
                np.array([np.mean(m) for m in max_90_mask]))
    
    def visualize_grid_cells(self, positions_list, bottleneck_list, prefix="grid"):










        print("开始Grid Cell可视化...")
        

        all_positions = np.concatenate([p.reshape(-1, 2) for p in positions_list], axis=0)
        bottleneck_size = bottleneck_list[0].shape[-1]
        all_bottleneck = np.concatenate([b.reshape(-1, bottleneck_size) for b in bottleneck_list], axis=0)
        
        print(f"可视化数据形状: 位置 {all_positions.shape}, 激活 {all_bottleneck.shape}")
        

        filename = f"{prefix}_grid_scores.pdf"
        scores_60, scores_90, mask_60, mask_90 = self.get_scores_and_plot(
            all_positions, all_bottleneck, filename)
        

        avg_gridness_60 = np.mean(scores_60)
        avg_gridness_90 = np.mean(scores_90)
        

        top_indices_60 = np.argsort(-scores_60)[:10]
        top_scores_60 = scores_60[top_indices_60]
        
        print("\n=== Grid Cell分析结果 ===")
        print(f"60度 Gridness 平均分数: {avg_gridness_60:.4f}")
        print(f"90度 Gridness 平均分数: {avg_gridness_90:.4f}")
        print(f"Top 10 神经元 (60度): {top_indices_60}")
        print(f"对应分数: {top_scores_60}")
        

        results_path = os.path.join(self.save_dir, f"{prefix}_grid_scores.txt")
        with open(results_path, "w") as f:
            f.write("=== Grid Cell分析结果 ===\n")
            f.write(f"60度 Gridness 平均分数: {avg_gridness_60:.4f}\n")
            f.write(f"90度 Gridness 平均分数: {avg_gridness_90:.4f}\n")
            f.write(f"Top 10 神经元 (60度): {top_indices_60}\n")
            f.write(f"对应分数: {top_scores_60}\n\n")
            
            f.write("所有神经元的得分:\n")
            for i in range(len(scores_60)):
                f.write(f"神经元 {i}: 60度分数 = {scores_60[i]:.4f}, 90度分数 = {scores_90[i]:.4f}\n")
        

        if np.any(scores_60 > 0.3):
            high_score_indices = np.where(scores_60 > 0.3)[0]
            if len(high_score_indices) >= 3:
                self.visualize_phase_relationship(high_score_indices[:3], all_positions, all_bottleneck, prefix)
        
        return avg_gridness_60
    
    def visualize_phase_relationship(self, rgb_neurons, positions, activations, prefix="grid"):









        ratemaps = []
        for neuron_idx in rgb_neurons:
            ratemap = self.scorer.calculate_ratemap(
                positions[:, 0], positions[:, 1], activations[:, neuron_idx])
            

            ratemap_min = np.nanmin(ratemap)
            ratemap_max = np.nanmax(ratemap)
            if ratemap_max > ratemap_min:
                normalized = (ratemap - ratemap_min) / (ratemap_max - ratemap_min)
            else:
                normalized = np.zeros_like(ratemap)
            ratemaps.append(normalized)
        

        rgb_map = np.stack([np.nan_to_num(r) for r in ratemaps], axis=-1)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_map, origin='lower', extent=[0, self.env_size, 0, self.env_size])
        plt.title(f"Grid Cell Phase Relationship (R: {rgb_neurons[0]}, G: {rgb_neurons[1]}, B: {rgb_neurons[2]})")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.colorbar(label="Normalized activation")
        plt.tight_layout()
        

        save_path = os.path.join(self.save_dir, f'{prefix}_phases.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存相位关系图到: {save_path}")


def load_model(model_path, device):









    try:

        checkpoint = torch.load(model_path, map_location=device)
        print(f"成功加载检查点，训练轮次: {checkpoint.get('epoch', 'unknown')}")
        

        from models.toroidal_grid_cell import GridCellNetwork
        from models.place_hd_cells import PlaceCellEnsemble, HeadDirectionCellEnsemble
        from config import Config
        

        config = Config()
        

        place_cells = PlaceCellEnsemble(
            n_cells=config.PLACE_CELLS_N,
            scale=config.PLACE_CELLS_SCALE,
            pos_min=0,
            pos_max=config.ENV_SIZE,
            seed=config.SEED
        )
        
        hd_cells = HeadDirectionCellEnsemble(
            n_cells=config.HD_CELLS_N,
            concentration=config.HD_CELLS_CONCENTRATION,
            seed=config.SEED
        )
        

        model = GridCellNetwork(
            place_cells=place_cells,
            hd_cells=hd_cells,
            input_size=3,
            hidden_size=config.HIDDEN_SIZE,
            bottleneck_size=256,
            dropout_rate=config.DROPOUT_RATE
        ).to(device)
        

        model_state_dict = checkpoint['model_state_dict']
        

        if all(k.startswith('module.') for k in model_state_dict.keys()):

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model_state_dict = new_state_dict
        
        model.load_state_dict(model_state_dict)
        print("模型参数加载成功")
        
        return model
    
    except Exception as e:
        print(f"加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_dataloader(split='val', batch_size=16, num_workers=2):










    try:

        from datasets.navigation_dataset import EnhancedNavigationDataset
        from config import Config
        

        config = Config()
        

        trajectory_folders = [
            d for d in os.listdir(config.DATA_ROOT)
            if os.path.isdir(os.path.join(config.DATA_ROOT, d))
            and d.startswith('D')
        ]
        trajectory_folders.sort()
        

        n_total = len(trajectory_folders)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        if split == 'train':
            folders = trajectory_folders[:n_train]
        elif split == 'val':
            folders = trajectory_folders[n_train:n_train + n_val]
        else:
            folders = trajectory_folders[n_train + n_val:]
        
        dataset_dirs = []
        for folder in folders:
            dataset_dirs.append(os.path.join(config.DATA_ROOT, folder, '000000'))
        

        dataset = EnhancedNavigationDataset(
            maze_dirs=dataset_dirs,
            sequence_length=config.SEQUENCE_LENGTH,
            stride=config.SEQUENCE_STRIDE,
            split=split
        )
        

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"创建{split}数据加载器成功，大小: {len(dataset)}，批次数: {len(dataloader)}")
        return dataloader
    
    except Exception as e:
        print(f"创建数据加载器时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Grid Cell可视化工具')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型的路径')
    parser.add_argument('--save_dir', type=str, default='grid_cell_viz', help='保存可视化结果的目录')
    parser.add_argument('--batch_size', type=int, default=32, help='数据批次大小')
    parser.add_argument('--num_batches', type=int, default=50, help='用于可视化的批次数量')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--prefix', type=str, default='grid', help='输出文件名前缀')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='使用的数据集分割')
    parser.add_argument('--nbins', type=int, default=40, help='Rate map分辨率')
    parser.add_argument('--env_size', type=float, default=15.0, help='环境大小')
    
    args = parser.parse_args()
    

    os.makedirs(args.save_dir, exist_ok=True)
    

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    

    model = load_model(args.model_path, device)
    if model is None:
        print("模型加载失败，退出程序")
        return
    

    dataloader = load_dataloader(split=args.split, batch_size=args.batch_size)
    if dataloader is None:
        print("数据加载器创建失败，退出程序")
        return
    

    visualizer = GridCellVisualizer(args.save_dir, nbins=args.nbins, env_size=args.env_size)
    

    positions_list, bottleneck_list = visualizer.collect_grid_cell_data(
        model, dataloader, device, num_batches=args.num_batches
    )
    

    avg_gridness = visualizer.visualize_grid_cells(positions_list, bottleneck_list, prefix=args.prefix)
    
    print(f"可视化完成！平均gridness分数: {avg_gridness:.3f}")
    print(f"结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
