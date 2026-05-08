"""
bipedalwalker_utils.py — BipedalWalker-v3 환경 유틸리티

reward_is_done_function: model rollout 전용 근사 보상.
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


def reward_is_done_function(next_obs: np.ndarray):
    """
    Model rollout용 근사 보상함수.
    Returns (reward, is_done) — 둘 다 (batch,) numpy array.

    계수 0.72 유도:
      gymnasium obs[2] = 0.3 * vel_x * (VIEWPORT_W/SCALE) / FPS = 0.12 * vel_x
      gymnasium shaping = 130 * pos_x / SCALE  →  Δshaping/step = 130/30 * (obs[2]/6) ≈ 0.722 * obs[2]
    fall penalty -100: 실제 환경 game_over 시 reward = -100
    """
    import torch
    if isinstance(next_obs, torch.Tensor):
        next_obs = next_obs.detach().cpu().numpy()
    next_obs = np.asarray(next_obs, dtype=np.float32)
    if next_obs.ndim == 1:
        next_obs = next_obs.reshape(1, -1)

    vel_x      = next_obs[:, 2]   # normalized horizontal velocity (obs[2] = 0.12 * real vel_x)
    hull_angle = next_obs[:, 0]   # hull angle (rad, 0 = upright)

    reward  = np.float32(0.72) * vel_x
    is_done = np.abs(hull_angle) > np.float32(1.4)
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
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def step_env(env, action) -> tuple:
    """Returns (next_obs, reward, done, info). Gymnasium 5-tuple 대응."""
    step_result = env.step(action)
    if len(step_result) == 5:
        next_obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return next_obs, reward, done, info
    if len(step_result) == 4:
        next_obs, reward, done, info = step_result
        return next_obs, reward, bool(done), info
    raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")
