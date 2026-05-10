"""
normalized_rbf.py — Normalized RBF Uncertainty Model for BipedalWalker
Input: [obs(24), act(4)] = 28D, Output: DOB_DIM=7 (uncertainty per velocity component)
Centers는 고정, Weights만 학습.
"""
import torch
import torch.nn as nn
from ..dynamics.constants import PHYS_MIN, PHYS_MAX, DOB_DIM


class NormalizedRBFModel(nn.Module):
    def __init__(self, num_centers: int = 600, width: float = 0.1,
                 initial_value: float = 5.0):
        super().__init__()
        phys_min_t = torch.tensor(PHYS_MIN)   # (28,)
        phys_max_t = torch.tensor(PHYS_MAX)   # (28,)
        in_dim     = len(PHYS_MIN)            # 28

        # Centers: [0, 1] 범위 uniform 샘플 후 raw 공간으로 역변환
        raw_centers = (phys_min_t.unsqueeze(1) +
                       (phys_max_t - phys_min_t).unsqueeze(1) *
                       torch.rand(in_dim, num_centers))           # (28, K)

        norm_centers = (2.0 * (raw_centers - phys_min_t.unsqueeze(1)) /
                        (phys_max_t - phys_min_t).unsqueeze(1) - 1.0)

        self.register_buffer('centers',  norm_centers)   # (28, K) — not trained
        self.register_buffer('phys_min', phys_min_t)
        self.register_buffer('phys_max', phys_max_t)
        self.width   = width
        self.weights = nn.Parameter(torch.ones(DOB_DIM, num_centers) * initial_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 28) — raw [obs, act]
        Returns (batch, 7) — uncertainty per velocity component
        """
        x_norm   = (2.0 * (x - self.phys_min) / (self.phys_max - self.phys_min) - 1.0)
        # ------------------------------------------------------------
        # Compute squared Euclidean distance between x and each center
        #
        # We use the identity:
        #   ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x^T c
        #
        # This allows efficient vectorized computation without explicit loops.
        #
        # Shapes:
        #   x_norm        : (batch, D)
        #   centers       : (D, K)
        #   result dist_sq: (batch, K)
        #
        # Meaning:
        #   dist_sq[b, i] = squared distance between x[b] and center c_i
        # ------------------------------------------------------------

        c_sq  = (self.centers ** 2).sum(dim=0)   # (K,)      -> ||c_i||^2
        x_sq  = (x_norm ** 2).sum(dim=1)         # (batch,)  -> ||x||^2
        cross = x_norm @ self.centers            # (batch,K) -> x^T c_i

        # Apply: ||x - c_i||^2 = ||x||^2 + ||c_i||^2 - 2 x^T c_i
        dist_sq = c_sq.unsqueeze(0) + x_sq.unsqueeze(1) - 2.0 * cross
        
        phi      = torch.exp(-dist_sq / (2.0 * self.width ** 2))
        phi_norm = phi / (phi.max(dim=1, keepdim=True).values + 1e-8)
        return phi_norm @ self.weights.t()                   # (batch, 7)
