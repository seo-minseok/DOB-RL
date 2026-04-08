"""
normalized_rbf.py — Normalized RBF Uncertainty Model
MATLAB: initNormalizedRBF_Structured + forwardNormalizedRBF
Only model.Weights are trainable; Centers are fixed at init.
"""
import torch
import torch.nn as nn
from ..dynamics.constants import PHYS_MIN, PHYS_MAX, FORCE_MAG


class NormalizedRBFModel(nn.Module):
    def __init__(self, num_centers: int = 600, width: float = 0.1,
                 initial_value: float = 5.0):
        super().__init__()
        phys_min_t = torch.tensor(PHYS_MIN)
        phys_max_t = torch.tensor(PHYS_MAX)

        # MATLAB: stateCenters = physMin(1:4) + (physMax(1:4)-physMin(1:4)).*rand(4,K)
        state_min   = torch.tensor(PHYS_MIN[:4]).unsqueeze(1)
        state_range = (torch.tensor(PHYS_MAX[:4]) - torch.tensor(PHYS_MIN[:4])).unsqueeze(1)
        state_centers = state_min + state_range * torch.rand(4, num_centers)

        half = num_centers // 2
        act_centers = torch.zeros(1, num_centers)
        act_centers[0, :half] = -float(FORCE_MAG)   # MATLAB: actCenters(1,1:half) = -10
        act_centers[0, half:] =  float(FORCE_MAG)   # MATLAB: actCenters(1,half+1:end) = 10

        raw_centers  = torch.cat([state_centers, act_centers], dim=0)   # (5, K)
        norm_centers = (2.0 * (raw_centers - phys_min_t.unsqueeze(1)) /
                        (phys_max_t - phys_min_t).unsqueeze(1) - 1.0)

        self.register_buffer('centers',  norm_centers)   # (5, K) — not trained
        self.register_buffer('phys_min', phys_min_t)
        self.register_buffer('phys_max', phys_max_t)
        self.width   = width
        self.weights = nn.Parameter(torch.ones(2, num_centers) * initial_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 5)  — raw [obs, act]
        Returns (batch, 2)  — predicted uncertainty
        """
        x_norm  = (2.0 * (x - self.phys_min) / (self.phys_max - self.phys_min) - 1.0)
        # MATLAB: distSq=(sum(C.^2,1)'+sum(xN.^2,1))-2*(C'*xN)  → (K,batch)
        c_sq    = (self.centers ** 2).sum(dim=0)          # (K,)
        x_sq    = (x_norm ** 2).sum(dim=1)                # (batch,)
        cross   = x_norm @ self.centers                    # (batch, K)
        dist_sq = c_sq.unsqueeze(0) + x_sq.unsqueeze(1) - 2.0 * cross  # (batch, K)
        phi     = torch.exp(-dist_sq / (2.0 * self.width ** 2))
        phi_norm = phi / (phi.sum(dim=1, keepdim=True) + 1e-8)
        return phi_norm @ self.weights.t()                 # (batch, 2)
