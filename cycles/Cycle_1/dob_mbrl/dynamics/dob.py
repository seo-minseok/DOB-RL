"""
dob.py — Disturbance Observer (DOB) 로직
MATLAB: predictNextObsDOB, DOB online update
"""
import numpy as np
import torch

from .constants import FPINV, F_MAT
from .nominal import step_nominal_cartpole


def predict_next_obs_dob(obs: np.ndarray, act: np.ndarray,
                          res_net, p_nom: dict,
                          use_nominal: bool) -> np.ndarray:
    """
    DOB 기반 다음 관측 예측.
    MATLAB: predictNextObsDOB
    nextObs = obs + dxNom + F * dxRes

    obs  : (batch, 4) numpy
    act  : (batch, 1) numpy  — continuous force
    Returns nextObs (batch, 4) numpy
    """
    if use_nominal:
        x_nom_next = step_nominal_cartpole(obs, act, p_nom)
        dx_nom     = x_nom_next - obs          # (batch, 4)
    else:
        dx_nom = np.zeros_like(obs)

    with torch.no_grad():
        inp    = torch.tensor(np.concatenate([obs, act], axis=-1))
        dx_res = res_net(inp).cpu().numpy()    # (batch, 2)

    next_obs = obs + dx_nom + (dx_res @ F_MAT.T)   # (batch, 4)
    return next_obs


def compute_dob_update(obs: np.ndarray, next_obs: np.ndarray,
                        action_force: float, dx_nom: np.ndarray,
                        res_net, dob_w: float, use_dob: bool) -> tuple:
    """
    DOB 온라인 업데이트.
    MATLAB: DOB.dhat = DOB.w*dxRes + (1-DOB.w)*Fpinv*e

    Returns (dhat, uncertainty) — 둘 다 (2,) numpy array.
    dhat은 에피소드마다 호출자가 zeros(2)로 리셋해야 함.
    """
    dx_real = next_obs - obs              # (4,)

    with torch.no_grad():
        inp_res = torch.tensor(
            np.concatenate([obs, [action_force]], dtype=np.float32)
        ).unsqueeze(0)
        dx_res = res_net(inp_res).cpu().numpy().flatten()   # (2,)

    e = dx_real - dx_nom                  # (4,)

    if use_dob:
        dhat = dob_w * dx_res + (1.0 - dob_w) * (FPINV @ e)
    else:
        dhat = np.zeros(2, dtype=np.float32)

    uncertainty = FPINV @ e - dx_res      # (2,)
    return dhat, uncertainty
