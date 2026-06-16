"""
q_network.py — SAC Critic for Hopper-v5
Input: obs(11D) + act(3D) = 14D,  Output: scalar Q-value
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, num_obs: int = 11, num_act: int = 3):
        super().__init__()
        in_dim = num_obs + num_act
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        obs : (batch, 11)
        act : (batch, 3)  ∈ [-1, 1]
        Returns (batch,) scalar Q-value
        """
        x = torch.cat([obs, act], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)   # (batch,)
