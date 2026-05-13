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

    x : (..., 24)  BipedalWalker observation
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
    # hip2_angle   += Ts * hip2_speed    (idx 8, 9  — contact 제거 후 재매핑)
    x_next[..., 8]  = x[..., 8]  + Ts * x[..., 9]
    # knee2_angle  += Ts * knee2_speed   (idx 10, 11 — contact 제거 후 재매핑)
    x_next[..., 10] = x[..., 10] + Ts * x[..., 11]
    # velocities (1,2,3,5,7,9,11) 그대로
    return x_next
