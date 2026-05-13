"""
residual_dx_net.py — Residual Dynamics Network for BipedalWalker
Input: [obs(24), act(4)] = 28D, Output: DOB_DIM=7 (velocity residuals)
"""
import torch
import torch.nn as nn
from ..dynamics.constants import IN_MIN, IN_MAX, DOB_DIM


class ResidualDxNet(nn.Module):
    def __init__(self, num_obs: int = 24, num_act: int = 4, hidden: int = 64):
        super().__init__()
        self.register_buffer('in_min', torch.tensor(IN_MIN))
        self.register_buffer('in_max', torch.tensor(IN_MAX))
        self.fc1 = nn.Linear(num_obs + num_act, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, DOB_DIM)   # 7D velocity residuals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 28)
        x_norm = 2.0 * (x - self.in_min) / (self.in_max - self.in_min) - 1.0
        x = torch.relu(self.fc1(x_norm))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)   # (batch, 7)
