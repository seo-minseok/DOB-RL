"""
constants.py — 전역 상수 (Gymnasium Hopper-v5, DOB 행렬)
"""
import numpy as np

# Observation: [z, root_angle, thigh, leg, foot, vx, vz, v_root, v_thigh, v_leg, v_foot]
OBS_DIM = 11
ACT_DIM = 3

OBS_MIN = np.array(
    [-0.5, -0.5, -1.0, -1.0, -1.0, -5.0, -5.0, -10.0, -10.0, -10.0, -10.0],
    dtype=np.float32,
)
OBS_MAX = np.array(
    [ 2.5,  0.5,  1.0,  1.0,  1.0,  5.0,  5.0,  10.0,  10.0,  10.0,  10.0],
    dtype=np.float32,
)

ACT_MIN = np.full(ACT_DIM, -1.0, dtype=np.float32)
ACT_MAX = np.full(ACT_DIM,  1.0, dtype=np.float32)

# DOB: 전체 state 추적 (DOB_DIM = OBS_DIM)
# FPINV @ e → e (identity),  F_MAT @ dx_res → dx_res (identity)
DOB_DIM = OBS_DIM
FPINV = np.eye(DOB_DIM, OBS_DIM, dtype=np.float32)   # (11, 11)
F_MAT = np.eye(OBS_DIM, DOB_DIM, dtype=np.float32)   # (11, 11)

# ResidualDxNet / NormalizedRBFModel 입력 범위: [obs(11D), act(3D)] = 14D
IN_MIN   = np.concatenate([OBS_MIN, ACT_MIN])
IN_MAX   = np.concatenate([OBS_MAX, ACT_MAX])
PHYS_MIN = IN_MIN
PHYS_MAX = IN_MAX

# Hopper-v5 건강 범위 (done 조건) — 공식 기본값 기준
# healthy_z_range:     [0.7, +∞)  → 상한 없음
# healthy_angle_range: [-0.2, 0.2]
# healthy_state_range: [-100, 100] for obs[1:]
HEALTHY_Z_MIN        = np.float32(0.7)
HEALTHY_ANG_MAX      = np.float32(0.2)
HEALTHY_STATE_MIN    = np.float32(-100.0)
HEALTHY_STATE_MAX    = np.float32(100.0)
