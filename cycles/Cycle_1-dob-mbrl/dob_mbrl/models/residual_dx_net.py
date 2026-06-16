"""
residual_dx_net.py — Residual Dynamics Network (Hopper-v5)
Input: [obs(11D), act(3D)] = 14-dim (raw, normalized internally)
Output: DOB_DIM-dim (11-dim state residual)
"""
import numpy as np
import torch
import torch.nn as nn


class ResidualDxNet(nn.Module):
    def __init__(self, num_obs: int = 11, num_act: int = 3,
                 hidden: int = 256, out_dim: int = 11):
        super().__init__()
        in_dim = num_obs + num_act
        self.fc1 = nn.Linear(in_dim,  hidden)
        self.fc2 = nn.Linear(hidden,  hidden)
        self.fc3 = nn.Linear(hidden,  hidden)
        self.fc4 = nn.Linear(hidden,  hidden)
        self.out = nn.Linear(hidden,  out_dim)

        # 입력 정규화용 running statistics (state_dict에 자동 포함)
        self.register_buffer('norm_mean', torch.zeros(in_dim))
        self.register_buffer('norm_std',  torch.ones(in_dim))

    def fit_normalizer(self, inputs: np.ndarray):
        """inputs: (N, in_dim) numpy array — real_buffer 학습 데이터 전체."""
        mean = inputs.mean(axis=0).astype(np.float32)
        std  = inputs.std(axis=0).astype(np.float32)
        std  = np.where(std < 1e-8, np.ones_like(std), std)
        self.norm_mean.copy_(torch.tensor(mean))
        self.norm_std.copy_(torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.out(x)
