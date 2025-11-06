




# /models/place_hd_cells.py

import torch
import numpy as np


class PlaceCellEnsemble:


    def __init__(self, n_cells, scale, pos_min=0, pos_max=15, seed=None):
        self.n_cells = n_cells
        rs = np.random.RandomState(seed)
        self.means = torch.from_numpy(
            rs.uniform(pos_min, pos_max, size=(self.n_cells, 2))
        ).float()
        self.variance = torch.ones_like(self.means) * (scale ** 2)

    def compute_activation(self, positions):
        """
        positions: [batch, seq, 2] or [batch, 2] ...
        return: same shape + [n_cells]
        """
        diff = positions.unsqueeze(-2) - self.means.to(positions.device)
        unnorm_logp = -0.5 * torch.sum((diff ** 2) / self.variance.to(positions.device), dim=-1)
        logits = unnorm_logp - torch.logsumexp(unnorm_logp, dim=-1, keepdim=True)
        return torch.exp(logits)


class HeadDirectionCellEnsemble:


    def __init__(self, n_cells, concentration, seed=None):
        self.n_cells = n_cells
        rs = np.random.RandomState(seed)
        self.means = torch.from_numpy(
            rs.uniform(-np.pi, np.pi, (n_cells,))
        ).float()
        self.kappa = torch.ones_like(self.means) * concentration

    def compute_activation(self, angles):
        """
        angles: [batch, seq] or [batch], in radians
        return: same shape + [n_cells]
        """
        diff = angles.unsqueeze(-1) - self.means.to(angles.device)
        unnorm_logp = self.kappa.to(angles.device) * torch.cos(diff)
        logits = unnorm_logp - torch.logsumexp(unnorm_logp, dim=-1, keepdim=True)
        return torch.exp(logits)
