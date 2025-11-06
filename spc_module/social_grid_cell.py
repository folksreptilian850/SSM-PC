import torch
import torch.nn as nn
from typing import Dict, Tuple

from grid_cell.models.place_hd_cells import (  # type: ignore
    HeadDirectionCellEnsemble,
    PlaceCellEnsemble,
)


class SocialGridCellNetwork(nn.Module):
    """Relational social navigation network with a shared bottleneck."""

    def __init__(self, place_cells: PlaceCellEnsemble, hd_cells: HeadDirectionCellEnsemble, config: Dict):
        super().__init__()
        self.place_cells = place_cells
        self.hd_cells = hd_cells
        self.hidden_size = config["HIDDEN_SIZE"]
        self.bottleneck_size = config["LATENT_DIM"]
        self.ego_token_size = config.get("ego_token_size", 4)

        self.ego_token = nn.Parameter(torch.randn(1, 1, self.ego_token_size))
        lstm_input_size = 3 + self.ego_token_size

        self.path_integrator_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        init_input_size = self.place_cells.n_cells + self.hd_cells.n_cells
        self.h0_generator = nn.Linear(init_input_size, self.hidden_size)
        self.c0_generator = nn.Linear(init_input_size, self.hidden_size)

        self.joint_bottleneck_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.bottleneck_size),
            nn.ReLU(),
            nn.Dropout(config.get("dropout_rate", 0.0)),
        )

        self.self_place_predictor = nn.Linear(self.bottleneck_size, self.place_cells.n_cells)
        self.self_hd_predictor = nn.Linear(self.bottleneck_size, self.hd_cells.n_cells)
        self.peer_place_predictor = nn.Linear(self.bottleneck_size, self.place_cells.n_cells)
        self.peer_hd_predictor = nn.Linear(self.bottleneck_size, self.hd_cells.n_cells)

        self.relational_head = nn.Sequential(
            nn.Linear(self.bottleneck_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _initial_state(self, init_pos: torch.Tensor, init_hd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        place_activation = self.place_cells.compute_activation(init_pos)
        hd_activation = self.hd_cells.compute_activation(init_hd)
        init_representation = torch.cat([place_activation, hd_activation], dim=-1)
        h0 = self.h0_generator(init_representation).unsqueeze(0)
        c0 = self.c0_generator(init_representation).unsqueeze(0)
        return h0, c0

    def forward(
        self,
        self_vel: torch.Tensor,
        self_init_pos: torch.Tensor,
        self_init_hd: torch.Tensor,
        peer_vel: torch.Tensor,
        peer_init_pos: torch.Tensor,
        peer_init_hd: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = self_vel.shape[:2]

        ego_token = self.ego_token.expand(batch_size, seq_len, -1)
        peer_token = torch.zeros_like(ego_token)
        self_vel_tagged = torch.cat([self_vel, ego_token], dim=-1)
        peer_vel_tagged = torch.cat([peer_vel, peer_token], dim=-1)

        h0_self, c0_self = self._initial_state(self_init_pos, self_init_hd)
        h0_peer, c0_peer = self._initial_state(peer_init_pos, peer_init_hd)

        lstm_out_self, _ = self.path_integrator_lstm(self_vel_tagged, (h0_self, c0_self))
        lstm_out_peer, _ = self.path_integrator_lstm(peer_vel_tagged, (h0_peer, c0_peer))
        joint_representation = lstm_out_self + lstm_out_peer

        flat_representation = joint_representation.reshape(batch_size * seq_len, self.hidden_size)
        bottleneck = self.joint_bottleneck_layer(flat_representation)

        self_place_logits = self.self_place_predictor(bottleneck)
        self_hd_logits = self.self_hd_predictor(bottleneck)
        peer_place_logits = self.peer_place_predictor(bottleneck)
        peer_hd_logits = self.peer_hd_predictor(bottleneck)
        predicted_distance = self.relational_head(bottleneck)

        def reshape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch_size, seq_len, -1)

        return {
            "self_place_logits": reshape(self_place_logits),
            "self_hd_logits": reshape(self_hd_logits),
            "peer_place_logits": reshape(peer_place_logits),
            "peer_hd_logits": reshape(peer_hd_logits),
            "predicted_distance": reshape(predicted_distance).squeeze(-1),
            "bottleneck_self": reshape(bottleneck),
        }
