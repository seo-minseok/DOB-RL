"""
cartpole_utils.py — CartPole 환경 유틸리티
MATLAB: rewardIsDoneFunction, makeCartpoleEnv, resetEnv, stepEnv
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

from ..dynamics.constants import X_THRESHOLD, THETA_THRESHOLD


def reward_is_done_function(next_obs: np.ndarray):
    """
    MATLAB: rewardIsDoneFunction
    Returns (reward, is_done) — 둘 다 (batch,) numpy array.
    """
    import torch
    if isinstance(next_obs, torch.Tensor):
        next_obs = next_obs.detach().cpu().numpy()
    next_obs = np.asarray(next_obs, dtype=np.float32)
    if next_obs.ndim == 1:
        next_obs = next_obs.reshape(1, -1)

    x     = next_obs[:, 0]
    theta = next_obs[:, 2]

    is_done    = (np.abs(x) > X_THRESHOLD) | (np.abs(theta) > THETA_THRESHOLD)
    r_angle    = 1.0 - (np.abs(theta) / THETA_THRESHOLD) ** 2
    r_pos      = 1.0 - (np.abs(x)     / X_THRESHOLD)     ** 2
    shaped_rew = 0.4 * 1.0 + 0.4 * r_angle + 0.2 * r_pos
    reward     = np.where(is_done, np.float32(-10.0), shaped_rew.astype(np.float32))
    return reward, is_done


def make_cartpole_env():
    if gym is None:
        raise ModuleNotFoundError(
            "Neither 'gymnasium' nor 'gym' is installed. "
            "Install with: python -m pip install gymnasium[classic-control]"
        )
    return gym.make('CartPole-v1')


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
