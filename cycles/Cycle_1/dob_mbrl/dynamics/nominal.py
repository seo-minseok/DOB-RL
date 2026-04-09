"""
nominal.py — CartPole Nominal Dynamics
MATLAB: defaultCartPoleParams, stepNominalCartPole
"""
import numpy as np
from .constants import FORCE_MAG


def default_cartpole_params() -> dict:
    return {
        'g': 9.8,
        'M': 1.0,
        'm': 0.1,
        'l': 0.5,
        'Ts': 0.02,
        'force_limit': float(FORCE_MAG),
    }


def step_nominal_cartpole(x: np.ndarray, u: np.ndarray, p: dict) -> np.ndarray:
    """
    Simplified Euler: xNomNext = x + Ts * [vel, 0, thd, 0]
    Accelerations are set to zero in the nominal model — DOB compensates.

    x : (..., 4)  [pos, vel, theta, thetadot]
    u : (..., 1)  force
    Returns xNomNext with same shape as x.
    """
    vel  = x[..., 1]
    thd  = x[..., 3]
    xdot = np.stack([vel,
                     np.zeros_like(vel),
                     thd,
                     np.zeros_like(thd)], axis=-1)
    return x + p['Ts'] * xdot
