"""
transition_network.py — BipedalWalker 전이 모델 (pure MBRL용)
train_MBRL_core.py 방식의 ensemble 전이 모델. DOB 없음.
Input: obs(14D) + act(4D) → delta obs(14D)
"""
import torch
import torch.nn as nn

from ..dynamics.constants import OBS_MIN, OBS_MAX


class TransitionNetwork(nn.Module):
    def __init__(self, num_obs: int = 14, num_act: int = 4, hidden: int = 256):
        super().__init__()
        self.register_buffer('obs_min', torch.tensor(OBS_MIN))
        self.register_buffer('obs_max', torch.tensor(OBS_MAX))

        self.fc1 = nn.Linear(num_obs + num_act, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_obs)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        obs : (batch, 14)
        act : (batch, 4) — already in [-1, 1], no normalization needed
        Returns delta obs (batch, 14)
        """
        obs_norm = 2.0 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1.0
        x = torch.cat([obs_norm, act], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
