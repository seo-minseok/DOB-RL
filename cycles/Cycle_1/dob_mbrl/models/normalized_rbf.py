"""
normalized_rbf.py — Normalized RBF Uncertainty Model
MATLAB: initNormalizedRBF_Structured + forwardNormalizedRBF
Only model.Weights are trainable; Centers are fixed at init.
"""
import torch
import torch.nn as nn
from ..dynamics.constants import PHYS_MIN, PHYS_MAX, DOB_DIM

_IN_DIM = len(PHYS_MIN)   # 14 (obs 11 + act 3)


class NormalizedRBFModel(nn.Module):
    def __init__(self, num_centers: int = 1200, width: float = 0.1,
                 initial_value: float = 5.0):
        super().__init__()
        phys_min_t = torch.tensor(PHYS_MIN)
        phys_max_t = torch.tensor(PHYS_MAX)

        raw_centers = (phys_min_t.unsqueeze(1) +
                       (phys_max_t - phys_min_t).unsqueeze(1) * torch.rand(_IN_DIM, num_centers))
        norm_centers = (2.0 * (raw_centers - phys_min_t.unsqueeze(1)) /
                        (phys_max_t - phys_min_t).unsqueeze(1) - 1.0)

        self.register_buffer('centers',  norm_centers)   # (_IN_DIM, K) — not trained
        self.register_buffer('phys_min', phys_min_t)
        self.register_buffer('phys_max', phys_max_t)
        self.width   = width
        self.weights = nn.Parameter(torch.ones(DOB_DIM, num_centers) * initial_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 14)  — raw [obs(11), act(3)]
        Returns (batch, DOB_DIM)  — predicted uncertainty
        """
        x_norm  = 2.0 * (x - self.phys_min) / (self.phys_max - self.phys_min) - 1.0
        c_sq    = (self.centers ** 2).sum(dim=0)          # (K,)
        x_sq    = (x_norm ** 2).sum(dim=1)                # (batch,)
        cross   = x_norm @ self.centers                    # (batch, K)
        dist_sq = c_sq.unsqueeze(0) + x_sq.unsqueeze(1) - 2.0 * cross  # (batch, K)
        phi     = torch.exp(-dist_sq / (2.0 * self.width ** 2))
        phi_norm = phi / (phi.sum(dim=1, keepdim=True) + 1e-8)
        return phi_norm @ self.weights.t()                 # (batch, DOB_DIM)
