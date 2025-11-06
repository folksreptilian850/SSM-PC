from __future__ import annotations

import numpy as np

from spc_module.config import SPCConfig


class TrajectoryGenerator:
    """Generate paired agent trajectories for four experimental conditions."""

    def __init__(self, config: SPCConfig):
        self.maze_size = config.ENV_SIZE
        self.seq_len = config.SEQUENCE_LENGTH
        self.start = np.array([self.maze_size * 0.5, self.maze_size * 0.2])
        self.goal_a = np.array([self.maze_size * 0.2, self.maze_size * 0.8])
        self.goal_b = np.array([self.maze_size * 0.8, self.maze_size * 0.8])

    def _smooth_path(self, start: np.ndarray, end: np.ndarray, moving: bool):
        if not moving:
            positions = np.tile(start, (self.seq_len, 1))
            return positions, np.zeros((self.seq_len, 2)), np.zeros(self.seq_len)

        base = np.array([start + (t / max(1, self.seq_len - 1)) * (end - start) for t in range(self.seq_len)])
        direction = (end - start) / (np.linalg.norm(end - start) + 1e-6)
        perpendicular = np.array([-direction[1], direction[0]])

        amplitude = np.random.uniform(0.5, 1.5)
        frequency = np.random.uniform(1.5, 2.5)
        phase = np.random.uniform(0, np.pi)
        t_axis = np.linspace(0, 1, self.seq_len)
        offset = amplitude * np.sin(2 * np.pi * frequency * t_axis + phase)

        positions = base + offset[:, None] * perpendicular[None, :]
        positions = np.clip(positions, 0.5, self.maze_size - 0.5)

        velocities = np.diff(positions, axis=0, prepend=positions[0:1])
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        angular_vels = np.diff(np.unwrap(angles), prepend=angles[0:1])
        return positions, velocities, angular_vels

    def sample(self, condition: int, num_reps: int):
        modes = {1: (True, False), 2: (False, True), 3: (True, True), 4: (False, False)}
        self_moving, peer_moving = modes[condition]

        self_pos, self_vel, self_ang = [], [], []
        peer_pos, peer_vel, peer_ang = [], [], []

        for i in range(num_reps):
            target = self.goal_a if i % 2 == 0 else self.goal_b
            static = self.start

            self_start, self_end = (self.start, target) if self_moving else (static, static)
            peer_start, peer_end = (self.start, target) if peer_moving else (static, static)

            s_pos, s_vel, s_ang = self._smooth_path(self_start, self_end, self_moving)
            p_pos, p_vel, p_ang = self._smooth_path(peer_start, peer_end, peer_moving)

            self_pos.append(s_pos)
            self_vel.append(s_vel)
            self_ang.append(s_ang)
            peer_pos.append(p_pos)
            peer_vel.append(p_vel)
            peer_ang.append(p_ang)

        return {
            "self_pos": np.array(self_pos),
            "self_vel": np.array(self_vel),
            "self_ang_vel": np.array(self_ang),
            "peer_pos": np.array(peer_pos),
            "peer_vel": np.array(peer_vel),
            "peer_ang_vel": np.array(peer_ang),
        }
