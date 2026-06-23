"""
dob.py — Disturbance Observer (DOB) 로직 (Hopper-v5)
DOB_DIM = 11 (전체 state 추적, FPINV = F_MAT = I_11)
"""
import numpy as np
import torch

from .constants import get_dob_matrices
from .nominal import step_nominal_hopper


def predict_next_obs_dob(obs: np.ndarray, act: np.ndarray,
                          res_net, p_nom: dict,
                          use_nominal: bool,
                          correct_position: bool = True) -> np.ndarray:
    """
    DOB 기반 다음 관측 예측.
    nextObs = obs + dx_nom + F_MAT @ dx_res

    obs              : (batch, 11) numpy
    act              : (batch, 3)  numpy  action in [-1, 1]
    correct_position : False → dob_dim=6, F_MAT(11,6)으로 velocity 슬롯에만 주입.
                       True  → dob_dim=11, F_MAT(11,11) identity로 전체 보정.
    Returns nextObs (batch, 11) numpy
    """
    dob_dim = 11 if correct_position else 6
    _, f_mat = get_dob_matrices(dob_dim)

    if use_nominal:
        x_nom_next = step_nominal_hopper(obs, act, p_nom)
        dx_nom     = x_nom_next - obs          # (batch, 11)
    else:
        dx_nom = np.zeros_like(obs)

    _device = next(res_net.parameters()).device
    with torch.no_grad():
        inp    = torch.tensor(np.concatenate([obs, act], axis=-1), device=_device)
        dx_res = res_net(inp).cpu().numpy()    # (batch, dob_dim)

    next_obs = obs + dx_nom + (dx_res @ f_mat.T)   # (batch, 11)
    return next_obs

