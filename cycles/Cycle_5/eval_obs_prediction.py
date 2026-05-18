"""
eval_obs_prediction.py — 테스트 에피소드에서 실제 obs vs 모델 예측 obs 비교
학습된 DOB-RL 모델(actor, res_net, contact_net)로 한 에피소드를 실행하고,
각 step에서 세 가지 next_obs를 기록한다:
  - actual    : 환경에서 받은 실제 next_obs
  - nominal   : nominal dynamics만 사용한 예측
  - full      : nominal + residual + contact_net 예측

OBS_DIM_NAMES의 각 차원마다 독립적인 figure 파일로 저장.

Usage:
  cd cycles/Cycle_5
  python eval_obs_prediction.py \
      --checkpoint checkpoints_real_ratio=0.2_uncert_thresh=0.3/Champion_Seed3_BestModel.pt \
      --seed 3
"""
import argparse
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from dob_mbrl.dynamics import (
    default_bipedalwalker_params,
    step_nominal_bipedalwalker,
    F_MAT,
)
from dob_mbrl.dynamics.constants import OBS_DIM, ACT_DIM, OBS_DIM_NAMES
from dob_mbrl.envs.bipedalwalker_utils import step_env
from dob_mbrl.models import ActorNetwork, ResidualDxNet, ContactNet, QNetwork

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
                        default='checkpoints_real_ratio=0.2_uncert_thresh=0.3/Champion_Seed3_BestModel.pt')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=1600)
    return parser.parse_args()


def _load_compat(model, state_dict: dict):
    """체크포인트와 모델의 buffer 크기가 다를 때도 안전하게 로드."""
    for key, val in state_dict.items():
        if key in model._buffers:
            model._buffers[key] = val.clone()
    model.load_state_dict(state_dict)


def save_figure(fig, path):
    assert len(fig.axes) == 1, f"subplot 금지: axes 수={len(fig.axes)}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_episode(actor, res_net, contact_net, p_nom, env, max_steps, seed,
                critic1=None, critic2=None, obs_slice=None):
    """
    obs_slice : 14D obs에서 actor/res_net/critic에 넣을 인덱스 배열.
                None이면 전체(14D) 사용. 12D 체크포인트는 contact 제거 슬라이스 전달.
    nominal dynamics / F_MAT 연산은 항상 14D obs 기준으로 수행.
    actual / predicted 기록은 obs_slice 적용 후의 차원으로 저장.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_14d = reset_render_env(env)   # 환경 obs는 항상 14D

    actual_list = []
    full_list   = []
    q1_list     = []
    q2_list     = []
    frames      = []

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame.astype(np.uint8))

        # 모델 입력용 obs (12D or 14D)
        obs_model = obs_14d if obs_slice is None else obs_14d[obs_slice]
        obs_t     = torch.tensor(obs_model).unsqueeze(0)
        act_t_inp = torch.tensor(np.concatenate([obs_model, np.zeros(ACT_DIM, np.float32)]))

        with torch.no_grad():
            action = actor(obs_t).cpu().numpy().flatten()

        next_obs_14d_actual, _, is_done, _ = step_env(env, action)

        # nominal dynamics: 항상 14D obs 사용 (인덱스 하드코딩 때문)
        next_obs_14d_nominal = step_nominal_bipedalwalker(
            obs_14d.reshape(1, -1), action.reshape(1, -1), p_nom
        ).flatten()
        dx_nom_14d = next_obs_14d_nominal - obs_14d   # (14,)

        # res_net: obs_model(12 or 14D) + action 입력
        inp   = torch.tensor(np.concatenate([obs_model, action], dtype=np.float32)).unsqueeze(0)
        act_t = torch.tensor(action).unsqueeze(0)
        with torch.no_grad():
            dx_res = res_net(inp).cpu().numpy().flatten()   # (7,)
            if critic1 is not None:
                q1_list.append(float(critic1(obs_t, act_t).item()))
            if critic2 is not None:
                q2_list.append(float(critic2(obs_t, act_t).item()))

        # 예측 next_obs: 14D 공간에서 계산
        next_obs_14d_pred = obs_14d + dx_nom_14d + (dx_res @ F_MAT.T)
        if contact_net is not None:
            with torch.no_grad():
                contact_pred = contact_net(inp).cpu().numpy().flatten()
            next_obs_14d_pred[8]  = contact_pred[0]
            next_obs_14d_pred[13] = contact_pred[1]

        # 저장: obs_slice 기준으로 축소
        if obs_slice is None:
            actual_list.append(next_obs_14d_actual.copy())
            full_list.append(next_obs_14d_pred.copy())
        else:
            actual_list.append(next_obs_14d_actual[obs_slice].copy())
            full_list.append(next_obs_14d_pred[obs_slice].copy())

        obs_14d = next_obs_14d_actual
        if is_done:
            break

    return (np.array(actual_list),
            np.array(full_list),
            np.array(q1_list),
            np.array(q2_list),
            frames)


def main():
    args = parse_args()

    # --- 모델 로드 ---
    ckpt = torch.load(args.checkpoint, weights_only=False)

    # 체크포인트에 저장된 obs_dim을 자동 감지 (버전 호환성)
    ckpt_obs_dim = ckpt['critic1']['obs_min'].shape[0]
    if ckpt_obs_dim != OBS_DIM:
        print(f'[eval] checkpoint obs_dim={ckpt_obs_dim} (current OBS_DIM={OBS_DIM})')

    # ckpt_obs_dim < OBS_DIM인 경우 contact 차원(8, 13)을 제거한 슬라이스 사용
    if ckpt_obs_dim == OBS_DIM:
        obs_slice = np.arange(OBS_DIM)
    else:
        obs_slice = np.array([i for i in range(OBS_DIM) if i not in (8, 13)], dtype=np.int32)
        assert len(obs_slice) == ckpt_obs_dim, \
            f'obs_slice 길이 {len(obs_slice)} ≠ ckpt_obs_dim {ckpt_obs_dim}'

    actor       = ActorNetwork(ckpt_obs_dim, ACT_DIM)
    res_net     = ResidualDxNet(ckpt_obs_dim, ACT_DIM, hidden=64)
    critic1     = QNetwork(ckpt_obs_dim, ACT_DIM)
    critic2     = QNetwork(ckpt_obs_dim, ACT_DIM)

    _load_compat(actor,   ckpt['actor'])
    _load_compat(res_net, ckpt['res_net'])
    _load_compat(critic1, ckpt['critic1'])
    _load_compat(critic2, ckpt['critic2'])

    contact_net = None
    if 'contact_net' in ckpt:
        contact_net = ContactNet(ckpt_obs_dim, ACT_DIM, hidden=64)
        _load_compat(contact_net, ckpt['contact_net'])
        contact_net.eval()

    actor.eval()
    res_net.eval()
    critic1.eval()
    critic2.eval()

    p_nom = default_bipedalwalker_params()
    env   = make_render_env()

    print(f'[eval] checkpoint: {args.checkpoint}')
    print(f'[eval] seed={args.seed}, max_steps={args.max_steps}')

    actual, full, q1_vals, q2_vals, frames = run_episode(
        actor, res_net, contact_net, p_nom, env, args.max_steps, args.seed,
        critic1=critic1, critic2=critic2, obs_slice=obs_slice,
    )
    env.close()

    T = actual.shape[0]
    steps = np.arange(T)
    print(f'[eval] episode length: {T} steps')

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'figures', 'obs_prediction'
    )

    # --- mp4 저장 ---
    video_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'figures', 'videos'
    )
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f'dob_rl_seed{args.seed}.mp4')
    if frames:
        imageio.mimwrite(video_path, frames, fps=50, macro_block_size=1)
        print(f'[eval] video saved → figures/videos/dob_rl_seed{args.seed}.mp4')

    # --- 차원별 figure 저장 ---
    dim_names_to_plot = (OBS_DIM_NAMES if obs_slice is None
                         else [OBS_DIM_NAMES[i] for i in obs_slice])
    for dim_idx, dim_name in enumerate(dim_names_to_plot):
        if dim_idx >= actual.shape[1]:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        ax.plot(steps, actual[:, dim_idx],
                color='tab:blue',  linewidth=1.2,
                label='actual')
        ax.plot(steps, full[:, dim_idx],
                color='tab:green', linewidth=1.0, linestyle='-.',
                label='nominal + residual + contact')

        ax.set_xlabel('step')
        ax.set_ylabel(dim_name)
        ax.set_title(f'obs prediction: {dim_name}  (seed={args.seed})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        fname = f'{dim_idx:02d}_{dim_name}.png'
        save_figure(fig, os.path.join(out_dir, fname))
        print(f'  saved → figures/obs_prediction/{fname}')

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
        print('  saved → figures/obs_prediction/q_function.png')

    print('[eval] done.')


if __name__ == '__main__':
    main()
