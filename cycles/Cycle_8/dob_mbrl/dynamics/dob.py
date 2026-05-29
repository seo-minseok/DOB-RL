"""
dob.py — Disturbance Observer (DOB) 로직 (Hopper-v5)
DOB_DIM = 11 (전체 state 추적, FPINV = F_MAT = I_11)
"""
import numpy as np
import torch

from .constants import FPINV, F_MAT, DOB_DIM
from .nominal import step_nominal_hopper


def predict_next_obs_dob(obs: np.ndarray, act: np.ndarray,
                          res_net, p_nom: dict,
                          use_nominal: bool) -> np.ndarray:
    """
    DOB 기반 다음 관측 예측.
    nextObs = obs + dx_nom + F_MAT @ dx_res

    obs  : (batch, 11) numpy
    act  : (batch, 3)  numpy  action in [-1, 1]
    Returns nextObs (batch, 11) numpy
    """
    if use_nominal:
        x_nom_next = step_nominal_hopper(obs, act, p_nom)
        dx_nom     = x_nom_next - obs          # (batch, 11) → zeros (null model)
    else:
        dx_nom = np.zeros_like(obs)

    with torch.no_grad():
        inp    = torch.tensor(np.concatenate([obs, act], axis=-1))
        dx_res = res_net(inp).cpu().numpy()    # (batch, 11)

    next_obs = obs + dx_nom + (dx_res @ F_MAT.T)   # (batch, 11)
    return next_obs


def compute_dob_update(obs: np.ndarray, next_obs: np.ndarray,
                        action: np.ndarray, dx_nom: np.ndarray,
                        res_net, dob_w: float, use_dob: bool) -> tuple:
    """
    DOB 온라인 업데이트.
    dhat = dob_w*dx_res + (1-dob_w)*FPINV*e

    Returns (dhat, uncertainty) — 둘 다 (DOB_DIM,) numpy.
    dhat은 에피소드마다 호출자가 zeros(DOB_DIM)으로 리셋해야 함.
    """
    dx_real = next_obs - obs              # (11,)

    with torch.no_grad():
        inp_res = torch.tensor(
            np.concatenate([obs, action], dtype=np.float32)
        ).unsqueeze(0)
        dx_res = res_net(inp_res).cpu().numpy().flatten()   # (11,)

    e = dx_real - dx_nom                  # (11,)

    if use_dob:
        dhat = dob_w * dx_res + (1.0 - dob_w) * (FPINV @ e)   # (11,)
    else:
        dhat = np.zeros(DOB_DIM, dtype=np.float32)

    uncertainty = FPINV @ e - dx_res      # (11,)
    return dhat, uncertainty
