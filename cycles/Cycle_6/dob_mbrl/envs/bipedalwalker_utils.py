"""
bipedalwalker_utils.py — BipedalWalker-v3 환경 유틸리티

reward_is_done_function: model rollout 전용 보상.
실제 환경 상호작용에서는 env.step()의 reward/done을 직접 사용할 것.

Cycle 5 obs 처리:
  raw 24D → lidar(14-23) 제거 → 14D (contact 포함)
  인덱스: [0,1,2,3,4,5,6,7,8,9,10,11,12,13] (원본 기준)
"""
import os
import numpy as np

# 원본 24D obs에서 lidar(14-23) 제거 후 14D 전체 유지 (contact 8, 13 포함)
_OBS_KEEP = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)

if os.name == 'nt' and os.environ.get('MUJOCO_GL', '').lower() == 'egl':
    os.environ.pop('MUJOCO_GL', None)

try:
    import gymnasium as gym
except ModuleNotFoundError:
    try:
        import gym
    except ModuleNotFoundError:
        gym = None

# Gymnasium BipedalWalker-v3 상수
_SCALE         = 30.0
_FPS           = 50
_MOTORS_TORQUE = 80

# 전진 속도 인센티브 계수 (제자리 서 있기 수렴 방지)
# obs[2] = 0.12 * vel_x (정규화된 수평 속도)
# - 서 있을 때 (obs[2]≈0): forward_incentive = -_FWD_BASELINE < 0  → 매 step 소폭 음수
# - 전진 시    (obs[2]=0.3): forward_incentive = 0.25              → 명확히 양수
_FWD_VEL_COEFF = np.float32(1.0)
_FWD_BASELINE  = np.float32(0.1)


def reward_is_done_function(obs: np.ndarray,
                             action: np.ndarray,
                             next_obs: np.ndarray):
    """
    Model rollout용 보상함수. Gymnasium BipedalWalker-v3 reward를 정교하게 재현.
    Returns (reward, is_done) — 둘 다 (batch,) numpy array.

    Gymnasium 공식:
      shaping   = 130/SCALE * pos_x - 5.0 * |hull_angle|
      reward    = Δshaping - 0.00035 * MOTORS_TORQUE * Σ clip(|a|, 0, 1)
      game_over → reward = -100

    pos_x 추정:
      obs[2] = 0.3 * vel_x * (VIEWPORT_W/SCALE) / FPS = 0.12 * vel_x
      Δpos_x ≈ vel_x / FPS = obs[2] / 6.0

    추가 항 (제자리 수렴 방지):
      forward_incentive = _FWD_VEL_COEFF * obs[2] - _FWD_BASELINE
      - obs[2] 비례항: 전진 속도를 직접 보상 (후진 패널티 포함)
      - baseline 패널티: vel_x=0 시 매 step -_FWD_BASELINE → 서 있기가 항상 음수

    is_done 조건 (실제 환경 game_over 근사):
      (1) |hull_angle| > 1.4               — 옆으로 넘어짐
      (2) vel_y < -0.3 AND |hull_angle|<0.5 — 수평 자세로 빠르게 주저앉기
      (3) both_contact AND vel_y < -0.15   — 양발 접지 상태에서 몸통 하강
    """
    import torch
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    obs      = _to_np(obs)
    action   = _to_np(action)
    next_obs = _to_np(next_obs)

    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    if next_obs.ndim == 1:
        next_obs = next_obs.reshape(1, -1)

    # Δshaping = 130/SCALE * Δpos_x - 5.0 * (|next_hull_angle| - |hull_angle|)
    delta_pos_x     = obs[:, 2] / np.float32(6.0)   # obs[2] / 6 ≈ Δpos_x
    d_pos_shaping   = (np.float32(130.0) / np.float32(_SCALE)) * delta_pos_x
    d_angle_shaping = np.float32(-5.0) * (np.abs(next_obs[:, 0]) - np.abs(obs[:, 0]))

    # motor cost: 0.00035 * MOTORS_TORQUE * Σ clip(|a|, 0, 1)
    motor_cost = (np.float32(0.00035) * np.float32(_MOTORS_TORQUE)
                  * np.sum(np.clip(np.abs(action), 0.0, 1.0), axis=-1))

    # 전진 속도 인센티브: 서 있기 수렴 방지
    # - obs[2] 항: 전진 시 양수, 후진 시 음수
    # - baseline: vel_x=0 이어도 매 step 소폭 음수로 만들어 제자리 균형점 파괴
    forward_incentive = _FWD_VEL_COEFF * obs[:, 2] - _FWD_BASELINE

    reward = d_pos_shaping + d_angle_shaping - motor_cost + forward_incentive

    # --- is_done 조건 ---
    # (1) hull 각도 초과: 기존 조건
    angle_fail = np.abs(next_obs[:, 0]) > np.float32(1.4)

    # (2) 수평 자세로 주저앉기: hull이 수평(|angle|<0.5)이면서 빠르게 하강
    #     obs[3] = 0.08 * vel_y; -0.3 ≈ vel_y = -3.75 m/s
    squat_fail = ((next_obs[:, 3] < np.float32(-0.3)) &
                  (np.abs(next_obs[:, 0]) < np.float32(0.5)))

    # (3) 양발 접지 + hull 하강: 두 발이 땅에 붙은 채 몸통이 가라앉는 상태
    #     obs[8]=left_contact, obs[13]=right_contact
    ground_collapse = ((next_obs[:, 8]  > np.float32(0.5)) &
                       (next_obs[:, 13] > np.float32(0.5)) &
                       (next_obs[:, 3]  < np.float32(-0.15)))

    is_done = angle_fail | squat_fail | ground_collapse
    reward  = np.where(is_done, np.float32(-100.0), reward)
    return reward.astype(np.float32), is_done


def make_bipedalwalker_env():
    if gym is None:
        raise ModuleNotFoundError(
            "Neither 'gymnasium' nor 'gym' is installed. "
            "Install with: python -m pip install gymnasium[box2d]"
        )
    return gym.make('BipedalWalker-v3')


def reset_env(env) -> np.ndarray:
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    return np.asarray(obs, dtype=np.float32)[_OBS_KEEP]  # lidar 제거 → 14D (contact 포함)


def step_env(env, action) -> tuple:
    """Returns (next_obs, reward, done, info). Gymnasium 5-tuple 대응."""
    step_result = env.step(action)
    if len(step_result) == 5:
        next_obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
    elif len(step_result) == 4:
        next_obs, reward, done, info = step_result
        done = bool(done)
    else:
        raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")
    return np.asarray(next_obs, dtype=np.float32)[_OBS_KEEP], reward, done, info  # lidar 제거 → 14D (contact 포함)
