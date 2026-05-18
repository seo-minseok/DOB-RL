"""
contact_net.py — Contact predictor for BipedalWalker
Input: obs(14D) + act(4D) = 18D → sigmoid(2) → [left_contact, right_contact]
Target: obs_{t+1} indices 8 (left_contact), 13 (right_contact)
"""
import torch
import torch.nn as nn
from ..dynamics.constants import IN_MIN, IN_MAX


class ContactNet(nn.Module):
    def __init__(self, num_obs: int = 14, num_act: int = 4, hidden: int = 64):
        super().__init__()
        self.register_buffer('in_min', torch.tensor(IN_MIN))
        self.register_buffer('in_max', torch.tensor(IN_MAX))
        self.fc1 = nn.Linear(num_obs + num_act, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 18) → (batch, 2) in [0, 1]"""
        x_norm = 2.0 * (x - self.in_min) / (self.in_max - self.in_min) - 1.0
        x = torch.relu(self.fc1(x_norm))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
