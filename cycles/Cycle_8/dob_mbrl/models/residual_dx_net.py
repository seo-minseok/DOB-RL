"""
residual_dx_net.py — Residual Dynamics Network (Hopper-v5)
Input: [obs(11D), act(3D)] = 14-dim
Output: DOB_DIM-dim (11-dim state residual)
"""
import torch
import torch.nn as nn
from ..dynamics.constants import IN_MIN, IN_MAX


class ResidualDxNet(nn.Module):
    def __init__(self, num_obs: int = 11, num_act: int = 3,
                 hidden: int = 64, out_dim: int = 11):
        super().__init__()
        self.register_buffer('in_min', torch.tensor(IN_MIN))
        self.register_buffer('in_max', torch.tensor(IN_MAX))
        self.fc1 = nn.Linear(num_obs + num_act, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 14)
        x_clamped = x.clamp(self.in_min, self.in_max)
        x_norm = 2.0 * (x_clamped - self.in_min) / (self.in_max - self.in_min) - 1.0
        x = torch.relu(self.fc1(x_norm))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)   # (batch, out_dim)
