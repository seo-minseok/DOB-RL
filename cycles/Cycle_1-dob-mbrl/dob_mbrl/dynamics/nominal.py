"""
nominal.py — Hopper-v5 Nominal Dynamics (kinematic integration)

Observation layout:
  obs[0:5]  = positions  [z, root_angle, thigh, leg, foot]
  obs[5]    = vx (forward velocity — no corresponding position in obs)
  obs[6:11] = velocities [vz, v_root, v_thigh, v_leg, v_foot]

Nominal model: Euler integration of positions, constant velocities.
  q_next  = q + qdot * dt     (obs[0:5] += obs[6:11] * dt)
  vx_next = vx                (obs[5] unchanged)
  v_next  = v                 (obs[6:11] unchanged — no force model)

DOB + ResidualDxNet가 실제 dynamics와 nominal의 차이(중력, contact, 토크 효과)를 보정.
dt = frame_skip * timestep = 4 * 0.002 = 0.008 s  (Hopper-v5 기본값)
"""
import numpy as np

DT = 0.008   # Hopper-v5: frame_skip=4, timestep=0.002


def default_hopper_params() -> dict:
    return {'dt': DT}


def step_nominal_hopper(x: np.ndarray, u: np.ndarray, p: dict) -> np.ndarray:
    """
    Kinematic nominal: positions integrated with current velocities.
    Velocities held constant (no force model).

    x : (..., 11)  Hopper observation
    u : (..., 3)   action in [-1, 1]  (unused — action-independent kinematics)
    Returns x_nom_next with same shape as x.
    """
    dt = p.get('dt', DT)
    x_next = x.copy()
    # obs[0:5] (positions) += obs[6:11] (velocities) * dt
    # index mapping: z↔obs[6], root↔obs[7], thigh↔obs[8], leg↔obs[9], foot↔obs[10]
    x_next[..., 0:5] = x[..., 0:5] + x[..., 6:11] * dt
    # obs[5] (vx) and obs[6:11] (velocities) unchanged
    return x_next
