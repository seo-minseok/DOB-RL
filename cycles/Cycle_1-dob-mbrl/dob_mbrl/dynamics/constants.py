"""
constants.py — 전역 상수 (Gymnasium Hopper-v5, DOB 행렬)
"""
import numpy as np

# Observation: [z, root_angle, thigh, leg, foot, vx, vz, v_root, v_thigh, v_leg, v_foot]
OBS_DIM = 11
ACT_DIM = 3

# 88,339 샘플(8 seeds, 125k steps) 실제 데이터 범위 기반 + 20% margin
# 구 PHYS 문제: thigh(-1.49 → clip at -1.0), leg(-2.17 → clip at -1.0) → 학습 불가
# Obs: [z, root_angle, thigh, leg, foot, vx, vz, v_root, v_thigh, v_leg, v_foot]
OBS_MIN = np.array(
    [ 0.5, -0.3, -1.8, -2.6, -1.3, -3.2, -6.1, -10.0, -10.0, -10.0, -10.0],
    dtype=np.float32,
)
OBS_MAX = np.array(
    [ 1.8,  0.3,  0.4,  0.6,  1.3,  5.3,  4.3,  10.0,  10.0,  10.0,  10.0],
    dtype=np.float32,
)

ACT_MIN = np.full(ACT_DIM, -1.0, dtype=np.float32)
ACT_MAX = np.full(ACT_DIM,  1.0, dtype=np.float32)

# DOB: 전체 state 추적 (DOB_DIM = OBS_DIM) — 기본값 (correct_position=True)
# FPINV @ e → e (identity),  F_MAT @ dx_res → dx_res (identity)
DOB_DIM = OBS_DIM
FPINV = np.eye(DOB_DIM, OBS_DIM, dtype=np.float32)   # (11, 11)
F_MAT = np.eye(OBS_DIM, DOB_DIM, dtype=np.float32)   # (11, 11)


def get_dob_matrices(dob_dim: int):
    """
    dob_dim=6  → correct_position=False: velocity-only (obs[5:11]), Cycle_1 호환
    dob_dim=11 → correct_position=True:  full state, Cycle_1-dob-mbrl 호환
    Returns (FPINV, F_MAT).
    """
    if dob_dim == 6:
        fpinv = np.zeros((6, OBS_DIM), dtype=np.float32)
        fpinv[:, 5:] = np.eye(6, dtype=np.float32)    # (6, 11): velocity 행만 추출
        f_mat = np.zeros((OBS_DIM, 6), dtype=np.float32)
        f_mat[5:, :] = np.eye(6, dtype=np.float32)    # (11, 6): velocity 행에만 주입
    else:
        fpinv = np.eye(dob_dim, OBS_DIM, dtype=np.float32)   # (11, 11) identity
        f_mat = np.eye(OBS_DIM, dob_dim, dtype=np.float32)   # (11, 11) identity
    return fpinv, f_mat

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
