"""
constants.py — BipedalWalker-v3 전역 상수 (관측/행동 경계, DOB 행렬)
Cycle 4: lidar(인덱스 14-23) 제거 → OBS_DIM=14
Cycle 5: lidar만 제거, leg contact(인덱스 8, 13) 복원 → OBS_DIM=14
"""
import numpy as np

OBS_DIM = 14  # lidar 제거(24→14), contact 포함
ACT_DIM = 4
DOB_DIM = 7   # DOB가 추적하는 velocity 성분 수

OBS_DIM_NAMES = [
    'hull_angle',   'hull_angvel',
    'vel_x',        'vel_y',
    'hip1_angle',   'hip1_speed',
    'knee1_angle',  'knee1_speed',
    'left_contact',
    'hip2_angle',   'hip2_speed',
    'knee2_angle',  'knee2_speed',
    'right_contact',
]

# Observation vector에서 velocity 성분 인덱스 (contact 포함 14D 기준)
# [hull_angvel(1), vel_x(2), vel_y(3), hip1_speed(5), knee1_speed(7), hip2_speed(10), knee2_speed(12)]
VELOCITY_INDICES = np.array([1, 2, 3, 5, 7, 10, 12], dtype=np.int32)

# BipedalWalker-v3 observation bounds (lidar 제외, contact 포함, 14D)
# 원본 인덱스 [0..13] 그대로 사용
OBS_MAX = np.array([
    np.pi,  # 0  hull angle
    5.0,    # 1  hull angular velocity
    5.0,    # 2  vel_x
    5.0,    # 3  vel_y
    np.pi,  # 4  hip1 joint angle
    np.pi,  # 5  hip1 joint speed
    np.pi,  # 6  knee1 joint angle
    np.pi,  # 7  knee1 joint speed
    1.0,    # 8  left_contact (binary 0/1)
    np.pi,  # 9  hip2 joint angle
    np.pi,  # 10 hip2 joint speed
    np.pi,  # 11 knee2 joint angle
    np.pi,  # 12 knee2 joint speed
    1.0,    # 13 right_contact (binary 0/1)
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
