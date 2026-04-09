"""
q_network.py — Q-Network (Critic)
MATLAB: featureInputLayer(rescale-symmetric) → FC(128)→ReLU→FC(128)→ReLU→FC(2)
"""
import torch
import torch.nn as nn
from ..dynamics.constants import OBS_MIN, OBS_MAX


class QNetwork(nn.Module):
    def __init__(self, num_observations: int = 4, num_actions: int = 2):
        super().__init__()
        self.register_buffer('obs_min', torch.tensor(OBS_MIN))
        self.register_buffer('obs_max', torch.tensor(OBS_MAX))
        self.fc1 = nn.Linear(num_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_norm = 2.0 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1.0
        x = torch.relu(self.fc1(obs_norm))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
