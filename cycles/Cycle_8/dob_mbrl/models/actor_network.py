"""
actor_network.py — DDPG Actor (Policy Network) for Hopper-v5
Input: obs 11D,  Output: action 3D, tanh bounded to [-1, 1]
"""
import torch
import torch.nn as nn
from ..dynamics.constants import OBS_MIN, OBS_MAX


class ActorNetwork(nn.Module):
    def __init__(self, num_obs: int = 11, num_act: int = 3):
        super().__init__()
        self.register_buffer('obs_min', torch.tensor(OBS_MIN))
        self.register_buffer('obs_max', torch.tensor(OBS_MAX))
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_act)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_clamped = obs.clamp(self.obs_min, self.obs_max)
        obs_norm = 2.0 * (obs_clamped - self.obs_min) / (self.obs_max - self.obs_min) - 1.0
        x = torch.relu(self.fc1(obs_norm))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))   # (batch, 3) ∈ [-1, 1]
