"""
q_network.py — TD3 Critic (Q-Network)
Input: obs (24D) + action (4D) = 28D, Output: scalar Q-value
CartPole의 discrete Q(s) → 연속 행동 Q(s, a) 로 전환.
"""
import torch
import torch.nn as nn
from ..dynamics.constants import OBS_MIN, OBS_MAX


class QNetwork(nn.Module):
    def __init__(self, num_obs: int = 24, num_act: int = 4):
        super().__init__()
        self.register_buffer('obs_min', torch.tensor(OBS_MIN))
        self.register_buffer('obs_max', torch.tensor(OBS_MAX))
        in_dim = num_obs + num_act
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        obs : (batch, 24)
        act : (batch, 4)   — action은 이미 [-1, 1] 범위이므로 정규화 불필요
        Returns (batch,) scalar Q-value
        """
        obs_norm = 2.0 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1.0
        x = torch.cat([obs_norm, act], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)   # (batch,)
