"""
bipedalwalker_utils.py — BipedalWalker-v3 환경 유틸리티

reward_is_done_function: model rollout 전용 보상.
실제 환경 상호작용에서는 env.step()의 reward/done을 직접 사용할 것.
"""
import os
import numpy as np

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
_SCALE        = 30.0
_FPS          = 50
_VIEWPORT_W   = 600
_MOTORS_TORQUE = 80


def reward_is_done_function(obs: np.ndarray,
                             action: np.ndarray,
                             next_obs: np.ndarray):
    """
    Model rollout용 보상함수. Gymnasium BipedalWalker-v3 reward를 재현.
    Returns (reward, is_done) — 둘 다 (batch,) numpy array.

    Gymnasium 공식:
      shaping   = 130/SCALE * pos_x - 5.0 * |hull_angle|
      reward    = Δshaping - 0.00035 * MOTORS_TORQUE * Σ clip(|a|, 0, 1)
      game_over → reward = -100

    pos_x 추정:
      obs[2] = 0.3 * vel_x * (VIEWPORT_W/SCALE) / FPS = 0.12 * vel_x
      Δpos_x ≈ vel_x / FPS = obs[2] / 6.0
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
    delta_pos_x      = obs[:, 2] / np.float32(6.0)   # obs[2] / 6 ≈ Δpos_x
    d_pos_shaping    = (np.float32(130.0) / np.float32(_SCALE)) * delta_pos_x
    d_angle_shaping  = np.float32(-5.0) * (np.abs(next_obs[:, 0]) - np.abs(obs[:, 0]))

    # motor cost: 0.00035 * MOTORS_TORQUE * Σ clip(|a|, 0, 1)
    motor_cost = (np.float32(0.00035) * np.float32(_MOTORS_TORQUE)
                  * np.sum(np.clip(np.abs(action), 0.0, 1.0), axis=-1))

    reward  = d_pos_shaping + d_angle_shaping - motor_cost

    # game_over 근사: 헐이 지면과 충돌하는 각도 (물리 시뮬레이션 없이 근사)
    is_done = np.abs(next_obs[:, 0]) > np.float32(1.4)
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
    return np.asarray(obs, dtype=np.float32)[:14]  # lidar(14-23) 제거


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
    return np.asarray(next_obs, dtype=np.float32)[:14], reward, done, info  # lidar 제거
