"""
nominal.py — BipedalWalker Nominal Dynamics (Kinematic model)
각도 = 이전 각도 + Ts * 각속도 (가속도 0 가정, DOB + residual이 보정)
"""
import numpy as np


def default_bipedalwalker_params() -> dict:
    return {'Ts': 0.02}


def step_nominal_bipedalwalker(x: np.ndarray, u: np.ndarray, p: dict) -> np.ndarray:
    """
    Kinematic nominal: position-like 상태들이 대응하는 velocity-like 상태로 적분.
    Zero acceleration 가정 — DOB와 residual net이 실제 동역학을 보정.

    x : (..., 14)  BipedalWalker observation (lidar 제거, contact 포함)
    u : (..., 4)   action (nominal에서 사용 안 함)
    Returns x_nom_next with same shape as x.
    """
    x_next = x.copy()
    Ts = p['Ts']
    # hull_angle   += Ts * hull_angvel   (idx 0, 1)
    x_next[..., 0]  = x[..., 0]  + Ts * x[..., 1]
    # hip1_angle   += Ts * hip1_speed    (idx 4, 5)
    x_next[..., 4]  = x[..., 4]  + Ts * x[..., 5]
    # knee1_angle  += Ts * knee1_speed   (idx 6, 7)
    x_next[..., 6]  = x[..., 6]  + Ts * x[..., 7]
    # left_contact (idx 8) — 그대로 복사 (x.copy()로 이미 처리됨)
    # hip2_angle   += Ts * hip2_speed    (idx 9, 10)
    x_next[..., 9]  = x[..., 9]  + Ts * x[..., 10]
    # knee2_angle  += Ts * knee2_speed   (idx 11, 12)
    x_next[..., 11] = x[..., 11] + Ts * x[..., 12]
    # right_contact (idx 13) — 그대로 복사 (x.copy()로 이미 처리됨)
    # velocities (1,2,3,5,7,10,12) 그대로
    return x_next
