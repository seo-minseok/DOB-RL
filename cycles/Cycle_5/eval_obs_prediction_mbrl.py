"""
eval_obs_prediction_mbrl.py — 테스트 에피소드에서 실제 obs vs MBRL 앙상블 예측 비교
학습된 MBRL 모델(actor, 3x TransitionNetwork)로 한 에피소드를 실행하고,
각 step에서 실제 next_obs와 앙상블 평균 예측 next_obs를 기록한다.

Usage:
  cd cycles/Cycle_5
  python eval_obs_prediction_mbrl.py \
      --checkpoint checkpoints/mbrl_real_ratio=0.2/MBRL_Seed3_BestModel.pt \
      --seed 3
"""
import argparse
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from dob_mbrl.dynamics.constants import OBS_DIM, ACT_DIM, OBS_DIM_NAMES
from dob_mbrl.envs.bipedalwalker_utils import step_env
from dob_mbrl.models import ActorNetwork, TransitionNetwork, QNetwork

_OBS_KEEP = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)


def make_render_env():
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        import gym
    return gym.make('BipedalWalker-v3', render_mode='rgb_array')


def reset_render_env(env) -> np.ndarray:
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    return np.asarray(obs, dtype=np.float32)[_OBS_KEEP]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/mbrl_real_ratio=0.2/MBRL_Seed3_BestModel.pt')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=1600)
    return parser.parse_args()


def save_figure(fig, path):
    assert len(fig.axes) == 1, f"subplot 금지: axes 수={len(fig.axes)}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_episode(actor, transition_models, env, max_steps, seed,
                critic1=None, critic2=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs = reset_render_env(env)

    actual_list = []
    mbrl_list   = []
    q1_list     = []
    q2_list     = []
    frames      = []

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame.astype(np.uint8))

        obs_t = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = actor(obs_t).cpu().numpy().flatten()

        next_obs_actual, _, is_done, _ = step_env(env, action)

        act_t = torch.tensor(action).unsqueeze(0)
        preds = []
        with torch.no_grad():
            for tm in transition_models:
                dx = tm(obs_t, act_t).cpu().numpy().flatten()
                preds.append(obs + dx)
            if critic1 is not None:
                q1_list.append(float(critic1(obs_t, act_t).item()))
            if critic2 is not None:
                q2_list.append(float(critic2(obs_t, act_t).item()))
        next_obs_mbrl = np.mean(preds, axis=0)

        actual_list.append(next_obs_actual.copy())
        mbrl_list.append(next_obs_mbrl.copy())

        obs = next_obs_actual
        if is_done:
            break

    return (np.array(actual_list),
            np.array(mbrl_list),
            np.array(q1_list),
            np.array(q2_list),
            frames)


def main():
    args = parse_args()

    # --- 모델 로드 ---
    ckpt = torch.load(args.checkpoint, weights_only=False)

    ckpt_obs_dim = ckpt['critic1']['obs_min'].shape[0]
    if ckpt_obs_dim != OBS_DIM:
        print(f'[eval_mbrl] checkpoint obs_dim={ckpt_obs_dim} (current OBS_DIM={OBS_DIM})')

    actor = ActorNetwork(ckpt_obs_dim, ACT_DIM)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    num_tm = 3
    transition_models = [TransitionNetwork(ckpt_obs_dim, ACT_DIM, hidden=256) for _ in range(num_tm)]
    for i, tm in enumerate(transition_models):
        tm.load_state_dict(ckpt[f'transition_model_{i}'])
        tm.eval()

    critic1 = QNetwork(ckpt_obs_dim, ACT_DIM)
    critic2 = QNetwork(ckpt_obs_dim, ACT_DIM)
    critic1.load_state_dict(ckpt['critic1'])
    critic2.load_state_dict(ckpt['critic2'])
    critic1.eval()
    critic2.eval()

    env = make_render_env()

    print(f'[eval_mbrl] checkpoint: {args.checkpoint}')
    print(f'[eval_mbrl] seed={args.seed}, max_steps={args.max_steps}')

    actual, mbrl, q1_vals, q2_vals, frames = run_episode(
        actor, transition_models, env, args.max_steps, args.seed,
        critic1=critic1, critic2=critic2,
    )
    env.close()

    T = actual.shape[0]
    steps = np.arange(T)
    print(f'[eval_mbrl] episode length: {T} steps')

    # --- mp4 저장 ---
    video_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'figures', 'videos'
    )
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f'mbrl_seed{args.seed}.mp4')
    if frames:
        imageio.mimwrite(video_path, frames, fps=50, macro_block_size=1)
        print(f'[eval_mbrl] video saved → figures/videos/mbrl_seed{args.seed}.mp4')

    # --- 차원별 figure 저장 ---
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'figures', 'obs_prediction_mbrl'
    )

    for dim_idx, dim_name in enumerate(OBS_DIM_NAMES):
        if dim_idx >= actual.shape[1]:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        ax.plot(steps, actual[:, dim_idx],
                color='tab:blue',   linewidth=1.2,
                label='actual')
        ax.plot(steps, mbrl[:, dim_idx],
                color='tab:orange', linewidth=1.0, linestyle='-.',
                label='MBRL ensemble (mean of 3)')

        ax.set_xlabel('step')
        ax.set_ylabel(dim_name)
        ax.set_title(f'obs prediction (MBRL): {dim_name}  (seed={args.seed})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        fname = f'{dim_idx:02d}_{dim_name}.png'
        save_figure(fig, os.path.join(out_dir, fname))
        print(f'  saved → figures/obs_prediction_mbrl/{fname}')

    # --- Q-function figure ---
    if len(q1_vals) > 0:
        q_steps = np.arange(len(q1_vals))
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        ax.plot(q_steps, q1_vals,
                color='tab:blue',  linewidth=1.2, label='Q1')
        ax.plot(q_steps, q2_vals,
                color='tab:orange', linewidth=1.2, label='Q2')
        ax.plot(q_steps, np.minimum(q1_vals, q2_vals),
                color='tab:red', linewidth=1.0, linestyle='--', label='min(Q1,Q2)')

        ax.set_xlabel('step')
        ax.set_ylabel('Q-value')
        ax.set_title(f'Q-function  Q(s_t, actor(s_t))  (seed={args.seed})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        save_figure(fig, os.path.join(out_dir, 'q_function.png'))
        print('  saved → figures/obs_prediction_mbrl/q_function.png')

    print('[eval_mbrl] done.')


if __name__ == '__main__':
    main()
