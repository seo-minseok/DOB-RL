"""
constants.py — 전역 상수 (CartPole 관측/행동 경계, DOB 행렬)
"""
import numpy as np

X_THRESHOLD                  = np.float32(2.4)
THETA_THRESHOLD              = np.float32(12.0 * np.pi / 180.0)
OBS_CART_POSITION_LIMIT      = np.float32(2.0 * X_THRESHOLD)
OBS_POLE_ANGLE_LIMIT         = np.float32(2.0 * THETA_THRESHOLD)
OBS_CART_VELOCITY_LIMIT      = np.float32(5.0)
OBS_POLE_ANGULAR_VELOCITY_LIMIT = np.float32(5.0)
FORCE_MAG                    = np.float32(10.0)

OBS_MIN = np.array([
    -OBS_CART_POSITION_LIMIT,
    -OBS_CART_VELOCITY_LIMIT,
    -OBS_POLE_ANGLE_LIMIT,
    -OBS_POLE_ANGULAR_VELOCITY_LIMIT,
], dtype=np.float32)

OBS_MAX = np.array([
    OBS_CART_POSITION_LIMIT,
    OBS_CART_VELOCITY_LIMIT,
    OBS_POLE_ANGLE_LIMIT,
    OBS_POLE_ANGULAR_VELOCITY_LIMIT,
], dtype=np.float32)

ACT_ELEMENTS = np.array([-FORCE_MAG, FORCE_MAG], dtype=np.float32)

# MATLAB: Fpinv = [0 1 0 0; 0 0 0 1]  — extracts velocity & theta_dot change
FPINV = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)

# MATLAB: F = [0 0; 1 0; 0 0; 0 1]  — maps 2D residual → 4D state update
F_MAT = np.array([[0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 1]], dtype=np.float32)

# ResidualDxNet / NormalizedRBFModel 입력 범위
IN_MIN  = np.array([*OBS_MIN, -FORCE_MAG], dtype=np.float32)
IN_MAX  = np.array([*OBS_MAX,  FORCE_MAG], dtype=np.float32)
PHYS_MIN = IN_MIN
PHYS_MAX = IN_MAX
