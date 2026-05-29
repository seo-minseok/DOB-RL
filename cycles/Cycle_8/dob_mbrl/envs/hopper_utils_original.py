"""
hopper_utils.py — Gymnasium Hopper-v5 환경 유틸리티

Observation (11D):
  [0]  z height
  [1]  root angle
  [2]  thigh joint angle
  [3]  leg joint angle
  [4]  foot joint angle
  [5]  x velocity (forward)
  [6]  z velocity
  [7]  root angular velocity
  [8]  thigh angular velocity
  [9]  leg angular velocity
  [10] foot angular velocity

Synthetic reward (model rollout 용):
  reward = forward_reward + healthy_reward - ctrl_cost
  forward_reward = (obs[...,5] + next_obs[...,5]) / 2   (trapezoid avg of x velocity)
  healthy_reward = 1.0 if healthy else 0.0
  ctrl_cost      = 0.001 * sum(action^2)
  healthy        = (z >= 0.7) and (|angle| <= 0.2)
                   and all(obs[1:] in [-100, 100])
  is_done        = ~healthy  (truncation은 외부 루프에서 처리)

  Note: gym computes forward_reward = (x_after - x_before) / dt,
        which ≈ (v_start + v_end) / 2  by trapezoidal integration.
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

from ..dynamics.constants import (
    HEALTHY_Z_MIN, HEALTHY_ANG_MAX, HEALTHY_STATE_MIN, HEALTHY_STATE_MAX,
)


def reward_is_done_function(obs: np.ndarray, action: np.ndarray,
                             next_obs: np.ndarray) -> tuple:
    """
    Synthetic reward for model rollout.
    Returns (reward, is_done) — 둘 다 (batch,) numpy array.

    obs      : (batch, 11)
    action   : (batch, 3)
    next_obs : (batch, 11)
    """
    import torch
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    if isinstance(next_obs, torch.Tensor):
        next_obs = next_obs.detach().cpu().numpy()
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    obs      = np.asarray(obs,      dtype=np.float32)
    next_obs = np.asarray(next_obs, dtype=np.float32)
    action   = np.asarray(action,   dtype=np.float32)
    if next_obs.ndim == 1:
        obs      = obs.reshape(1, -1)
        next_obs = next_obs.reshape(1, -1)
        action   = action.reshape(1, -1)

    z     = next_obs[:, 0]
    angle = next_obs[:, 1]

    state_healthy = np.all(
        (next_obs[:, 1:] >= float(HEALTHY_STATE_MIN)) &
        (next_obs[:, 1:] <= float(HEALTHY_STATE_MAX)),
        axis=1,
    )
    is_healthy = (
        state_healthy &
        (z >= float(HEALTHY_Z_MIN)) &
        (np.abs(angle) <= float(HEALTHY_ANG_MAX))
    )
    is_done    = ~is_healthy

    # gym: forward_reward = (x_after - x_before)/dt ≈ (vx_before + vx_after)/2
    forward_reward = (obs[:, 5] + next_obs[:, 5]) * 0.5
    ctrl_cost      = 0.001 * np.sum(np.clip(action, -1.0, 1.0) ** 2, axis=1)
    healthy_reward = np.where(is_healthy, np.float32(1.0), np.float32(0.0))

    reward = forward_reward + healthy_reward - ctrl_cost
    return reward.astype(np.float32), is_done


def make_hopper_env():
    if gym is None:
        raise ModuleNotFoundError(
            "Neither 'gymnasium' nor 'gym' is installed. "
            "Install with: python -m pip install gymnasium[mujoco]"
        )
    return gym.make('Hopper-v5')


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
