"""
dob.py — Disturbance Observer (DOB) for BipedalWalker
obs(24), act(4), DOB dim = 7 (velocity 성분)
"""
import numpy as np
import torch

from .constants import FPINV, F_MAT, DOB_DIM
from .nominal import step_nominal_bipedalwalker


def predict_next_obs_dob(obs: np.ndarray, act: np.ndarray,
                          res_net, p_nom: dict,
                          use_nominal: bool) -> np.ndarray:
    """
    DOB 기반 다음 관측 예측.
    nextObs = obs + dxNom + F_MAT * dxRes

    obs  : (batch, 24) numpy
    act  : (batch, 4)  numpy — continuous action
    Returns nextObs (batch, 24) numpy
    """
    if use_nominal:
        x_nom_next = step_nominal_bipedalwalker(obs, act, p_nom)
        dx_nom     = x_nom_next - obs          # (batch, 24)
    else:
        dx_nom = np.zeros_like(obs)

    with torch.no_grad():
        inp    = torch.tensor(np.concatenate([obs, act], axis=-1))
        dx_res = res_net(inp).cpu().numpy()    # (batch, 7)

    next_obs = obs + dx_nom + (dx_res @ F_MAT.T)   # (batch, 24)
    return next_obs


def compute_dob_update(obs: np.ndarray, next_obs: np.ndarray,
                        action: np.ndarray, dx_nom: np.ndarray,
                        res_net, dob_w: float, use_dob: bool) -> tuple:
    """
    DOB 온라인 업데이트.
    dhat = dob_w * dxRes + (1 - dob_w) * FPINV * e

    obs      : (24,) numpy
    next_obs : (24,) numpy
    action   : (4,)  numpy (continuous)
    dx_nom   : (24,) numpy
    Returns (dhat, uncertainty) — 둘 다 (7,) numpy array.
    dhat은 에피소드마다 호출자가 zeros(DOB_DIM)으로 리셋해야 함.
    """
    dx_real = next_obs - obs              # (24,)

    with torch.no_grad():
        inp_res = torch.tensor(
            np.concatenate([obs, action], dtype=np.float32)
        ).unsqueeze(0)
        dx_res = res_net(inp_res).cpu().numpy().flatten()   # (7,)

    e = dx_real - dx_nom                  # (24,)

    if use_dob:
        dhat = dob_w * dx_res + (1.0 - dob_w) * (FPINV @ e)
    else:
        dhat = np.zeros(DOB_DIM, dtype=np.float32)

    uncertainty = FPINV @ e - dx_res      # (7,)
    return dhat, uncertainty
