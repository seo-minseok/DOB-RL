"""
actor_network.py — SAC Actor (Stochastic Policy Network) for Hopper-v5
Input: obs 11D (raw, no normalization),  Output: action 3D via tanh-squashed Gaussian
forward() : deterministic mean (평가용)
sample()  : reparameterized stochastic sample + log_prob (학습용)
"""
import torch
import torch.nn as nn

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class ActorNetwork(nn.Module):
    def __init__(self, num_obs: int = 11, num_act: int = 3):
        super().__init__()
        self.fc1           = nn.Linear(num_obs, 256)
        self.fc2           = nn.Linear(256, 256)
        self.mean_layer    = nn.Linear(256, num_act)
        self.log_std_layer = nn.Linear(256, num_act)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(obs))
        return torch.relu(self.fc2(x))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic mean action — evaluation only."""
        return torch.tanh(self.mean_layer(self._features(obs)))

    def sample(self, obs: torch.Tensor):
        """
        Reparameterized stochastic sample with tanh-squashing correction.
        Returns: (action, log_prob)  shapes: (B, num_act), (B, 1)
        """
        x       = self._features(obs)
        mean    = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        dist    = torch.distributions.Normal(mean, std)
        z       = dist.rsample()
        action  = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)   # (B, 1)
        return action, log_prob
