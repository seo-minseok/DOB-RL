"""
constants.py — BipedalWalker-v3 전역 상수 (관측/행동 경계, DOB 행렬)
Cycle 4: lidar(인덱스 14-23) 제거 → OBS_DIM=14
"""
import numpy as np

OBS_DIM = 14  # lidar 10채널 제거 (24 → 14)
ACT_DIM = 4
DOB_DIM = 7   # DOB가 추적하는 velocity 성분 수

# Observation vector에서 velocity 성분 인덱스 (lidar 제거 후에도 동일)
# [hull_angvel(1), vel_x(2), vel_y(3), hip1_speed(5), knee1_speed(7), hip2_speed(10), knee2_speed(12)]
VELOCITY_INDICES = np.array([1, 2, 3, 5, 7, 10, 12], dtype=np.int32)

# BipedalWalker-v3 observation bounds (lidar 제외)
OBS_MAX = np.array([
    np.pi,  # 0  hull angle
    5.0,    # 1  hull angular velocity
    5.0,    # 2  vel_x
    5.0,    # 3  vel_y
    np.pi,  # 4  hip1 joint angle
    np.pi,  # 5  hip1 joint speed
    np.pi,  # 6  knee1 joint angle
    np.pi,  # 7  knee1 joint speed
    1.0,    # 8  leg1 ground contact
    np.pi,  # 9  hip2 joint angle
    np.pi,  # 10 hip2 joint speed
    np.pi,  # 11 knee2 joint angle
    np.pi,  # 12 knee2 joint speed
    1.0,    # 13 leg2 ground contact
], dtype=np.float32)
OBS_MIN = -OBS_MAX.copy()

# Action bounds (4 joint motors, each in [-1, 1])
ACT_MAX = np.ones(ACT_DIM, dtype=np.float32)
ACT_MIN = -ACT_MAX.copy()

# FPINV: (7, 14) — obs에서 velocity 성분 추출
FPINV = np.zeros((DOB_DIM, OBS_DIM), dtype=np.float32)
for _i, _vi in enumerate(VELOCITY_INDICES):
    FPINV[_i, _vi] = 1.0

# F_MAT: (14, 7) — DOB 보정을 obs 공간으로 역투영
F_MAT = FPINV.T.copy()

# ResidualDxNet / NormalizedRBFModel 입력 범위 (18 = 14 obs + 4 act)
IN_MIN  = np.array([*OBS_MIN, *ACT_MIN], dtype=np.float32)
IN_MAX  = np.array([*OBS_MAX, *ACT_MAX], dtype=np.float32)
PHYS_MIN = IN_MIN
PHYS_MAX = IN_MAX
