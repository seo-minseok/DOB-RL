"""
residual_dx_net.py — Residual Dynamics Network
MATLAB: initResidualDxNet(nObs=4, nAct=1, hidden=32)
Input: [obs, act] = 5-dim,  Output: 2-dim (vel & theta_dot residuals)
"""
import torch
import torch.nn as nn
from ..dynamics.constants import IN_MIN, IN_MAX


class ResidualDxNet(nn.Module):
    def __init__(self, num_obs: int = 4, num_act: int = 1, hidden: int = 32):
        super().__init__()
        self.register_buffer('in_min', torch.tensor(IN_MIN))
        self.register_buffer('in_max', torch.tensor(IN_MAX))
        self.fc1 = nn.Linear(num_obs + num_act, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)   # MATLAB: fullyConnectedLayer(2,"Name","dx_res")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 5)
        x_norm = 2.0 * (x - self.in_min) / (self.in_max - self.in_min) - 1.0
        x = torch.relu(self.fc1(x_norm))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)   # (batch, 2)
