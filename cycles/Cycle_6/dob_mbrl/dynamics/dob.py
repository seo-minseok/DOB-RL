"""
dob.py — Disturbance Observer (DOB) for BipedalWalker
obs(14), act(4), DOB dim = 7 (velocity 성분)
"""
import numpy as np
import torch

from .constants import FPINV, F_MAT, DOB_DIM
from .nominal import step_nominal_bipedalwalker


def predict_next_obs_dob(obs: np.ndarray, act: np.ndarray,
                          res_net, p_nom: dict,
                          use_nominal: bool,
                          contact_net=None) -> np.ndarray:
    """
    DOB 기반 다음 관측 예측.
    nextObs = obs + dxNom + F_MAT * dxRes
    contact_net이 주어지면 indices 8, 13을 ContactNet 출력으로 교체.

    obs  : (batch, 14) numpy
    act  : (batch, 4)  numpy — continuous action
    Returns nextObs (batch, 14) numpy
    """
    if use_nominal:
        x_nom_next = step_nominal_bipedalwalker(obs, act, p_nom)
        dx_nom     = x_nom_next - obs          # (batch, 14)
    else:
        dx_nom = np.zeros_like(obs)

    inp = torch.tensor(np.concatenate([obs, act], axis=-1))
    with torch.no_grad():
        dx_res = res_net(inp).cpu().numpy()    # (batch, 7)

    next_obs = obs + dx_nom + (dx_res @ F_MAT.T)   # (batch, 14)

    if contact_net is not None:
        with torch.no_grad():
            contact_pred = contact_net(inp).cpu().numpy()   # (batch, 2)
        next_obs[:, 8]  = contact_pred[:, 0]   # left_contact
        next_obs[:, 13] = contact_pred[:, 1]   # right_contact

    return next_obs


def compute_dob_update(obs: np.ndarray, next_obs: np.ndarray,
                        action: np.ndarray, dx_nom: np.ndarray,
                        res_net, dob_w: float, use_dob: bool) -> tuple:
    """
    DOB 온라인 업데이트.
    dhat = dob_w * dxRes + (1 - dob_w) * FPINV * e

    obs      : (14,) numpy
    next_obs : (14,) numpy
    action   : (4,)  numpy (continuous)
    dx_nom   : (14,) numpy
    Returns (dhat, uncertainty) — 둘 다 (7,) numpy array.
    dhat은 에피소드마다 호출자가 zeros(DOB_DIM)으로 리셋해야 함.
    """
    dx_real = next_obs - obs              # (14,)

    with torch.no_grad():
        inp_res = torch.tensor(
            np.concatenate([obs, action], dtype=np.float32)
        ).unsqueeze(0)
        dx_res = res_net(inp_res).cpu().numpy().flatten()   # (7,)

    e = dx_real - dx_nom                  # (14,)

    if use_dob:
        dhat = dob_w * dx_res + (1.0 - dob_w) * (FPINV @ e)
    else:
        dhat = np.zeros(DOB_DIM, dtype=np.float32)

    uncertainty = FPINV @ e - dx_res      # (7,)
    return dhat, uncertainty
