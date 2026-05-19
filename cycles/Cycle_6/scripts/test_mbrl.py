"""
test_mbrl.py — MBRL 학습된 모델 테스트 및 분석

저장 파일 (figures/test_mbrl/ckpt{ckpt_seed}_test{test_seed}/ 하위):
  reward.png               - 스텝별 reward + 누적 reward
  q_value.png              - 스텝별 Q1, Q2 값
  action_{name}.png        - 스텝별 행동 (4개)
  obs_pred_{name}.png      - 차원별 actual / 각 ensemble 모델 예측 비교 (14개)
  obs_pred_err_{name}.png  - 차원별 예측 오차 (14개)
  step_data.csv            - 모든 스텝 데이터
  episode.mp4              - 에피소드 영상

Usage:
  python scripts/test_mbrl.py --checkpoint-seed 1 --test-seed 42
  python scripts/test_mbrl.py --checkpoint ./checkpoints/mbrl_real_ratio=0.2/MBRL_Seed1_BestModel.pt --test-seed 42
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_CYCLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CYCLE_DIR not in sys.path:
    sys.path.insert(0, _CYCLE_DIR)

from dob_mbrl.models import ActorNetwork, QNetwork, TransitionNetwork
from dob_mbrl.dynamics import OBS_DIM, ACT_DIM, OBS_DIM_NAMES
from dob_mbrl.envs.bipedalwalker_utils import reset_env, step_env

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

ACT_NAMES = ['hip1_motor', 'knee1_motor', 'hip2_motor', 'knee2_motor']

# 앙상블 모델별 색상
_MODEL_COLORS = ['darkorange', 'green', 'purple', 'brown', 'crimson']


# ---------------------------------------------------------------------------
# Checkpoint 로드
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    actor = ActorNetwork(OBS_DIM, ACT_DIM)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    critic1 = QNetwork(OBS_DIM, ACT_DIM)
    critic1.load_state_dict(ckpt['critic1'])
    critic1.eval()

    critic2 = QNetwork(OBS_DIM, ACT_DIM)
    critic2.load_state_dict(ckpt['critic2'])
    critic2.eval()

    # transition_model_0, 1, 2 ... 있는 만큼 로드
    transition_models = []
    i = 0
    while f'transition_model_{i}' in ckpt:
        tm = TransitionNetwork(OBS_DIM, ACT_DIM, hidden=256)
        tm.load_state_dict(ckpt[f'transition_model_{i}'])
        tm.eval()
        transition_models.append(tm)
        i += 1

    num_tm = len(transition_models)
    print(f'[test_mbrl] Loaded checkpoint: {ckpt_path}')
    print(f'  episode={ckpt.get("episode", "?")}  total_steps={ckpt.get("total_steps", "?")}  num_models={num_tm}')
    return actor, critic1, critic2, transition_models


# ---------------------------------------------------------------------------
# 에피소드 실행
# ---------------------------------------------------------------------------

def run_episode(env, actor, critic1, critic2, transition_models):
    """
    한 에피소드를 실행하면서 모든 분석 데이터를 수집한다.

    Returns dict with keys:
      obs, action, next_obs_real,
      next_obs_per_model  : list of (T, OBS_DIM) — 모델별 예측
      next_obs_ensemble   : (T, OBS_DIM) — 앙상블 평균 예측
      reward, q1, q2,
      frames
    """
    obs = reset_env(env)
    frames = []

    obs_list             = []
    act_list             = []
    next_obs_real_list   = []
    next_obs_per_model   = [[] for _ in transition_models]
    reward_list          = []
    q1_list              = []
    q2_list              = []

    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_t = actor(obs_t)
            q1_val   = critic1(obs_t, action_t).item()
            q2_val   = critic2(obs_t, action_t).item()

        action = action_t.squeeze(0).numpy()
        act_t  = action_t  # (1, ACT_DIM)

        # 각 transition model 예측
        with torch.no_grad():
            for mi, tm in enumerate(transition_models):
                dx        = tm(obs_t, act_t).squeeze(0).numpy()
                pred_next = obs + dx
                next_obs_per_model[mi].append(pred_next.copy())

        next_obs, reward, done, _ = step_env(env, action)

        obs_list.append(obs.copy())
        act_list.append(action.copy())
        next_obs_real_list.append(next_obs.copy())
        reward_list.append(float(reward))
        q1_list.append(float(q1_val))
        q2_list.append(float(q2_val))

        obs = next_obs

    T = len(reward_list)
    next_obs_per_model_np = [np.array(m) for m in next_obs_per_model]  # list of (T, OBS_DIM)
    next_obs_ensemble     = np.mean(next_obs_per_model_np, axis=0)      # (T, OBS_DIM)

    return {
        'obs':               np.array(obs_list),
        'action':            np.array(act_list),
        'next_obs_real':     np.array(next_obs_real_list),
        'next_obs_per_model': next_obs_per_model_np,
        'next_obs_ensemble': next_obs_ensemble,
        'reward':            np.array(reward_list),
        'q1':                np.array(q1_list),
        'q2':                np.array(q2_list),
        'frames':            frames,
    }


# ---------------------------------------------------------------------------
# 저장 유틸
# ---------------------------------------------------------------------------

def _save_fig(fig, path):
    ax_list = fig.get_axes()
    assert len(ax_list) == 1, f"Figure must have exactly 1 Axes (got {len(ax_list)})"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def save_reward(reward, out_dir):
    steps  = np.arange(1, len(reward) + 1)
    cumrew = np.cumsum(reward)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='white')
    ax.plot(steps, reward,  color='steelblue', linewidth=1.2, label='Step reward', alpha=0.7)
    ax.plot(steps, cumrew,  color='darkorange', linewidth=1.8, label='Cumulative reward')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title(f'Reward  (total={cumrew[-1]:.1f})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.4)
    _save_fig(fig, os.path.join(out_dir, 'reward.png'))


def save_q_value(q1, q2, out_dir):
    steps = np.arange(1, len(q1) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='white')
    ax.plot(steps, q1, color='steelblue',  linewidth=1.5, label='Q1')
    ax.plot(steps, q2, color='darkorange', linewidth=1.5, label='Q2', linestyle='--')
    ax.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Q Value', fontsize=12, fontweight='bold')
    ax.set_title('Q-Function Values', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.4)
    _save_fig(fig, os.path.join(out_dir, 'q_value.png'))


def save_actions(action, out_dir):
    steps = np.arange(1, len(action) + 1)
    for i, name in enumerate(ACT_NAMES):
        fig, ax = plt.subplots(1, 1, figsize=(10, 3), facecolor='white')
        ax.plot(steps, action[:, i], color='steelblue', linewidth=1.3)
        ax.axhline( 1.0, color='red', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.axhline(-1.0, color='red', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Action', fontsize=12, fontweight='bold')
        ax.set_title(f'Action - {name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.4)
        _save_fig(fig, os.path.join(out_dir, f'action_{name}.png'))


def save_obs_predictions(next_obs_real, next_obs_per_model, next_obs_ensemble, out_dir):
    steps   = np.arange(1, len(next_obs_real) + 1)
    num_tm  = len(next_obs_per_model)

    for i, name in enumerate(OBS_DIM_NAMES):
        real     = next_obs_real[:, i]
        ensemble = next_obs_ensemble[:, i]

        # --- 예측 vs 실제 ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='white')
        ax.plot(steps, real, color='steelblue', linewidth=1.8, label='Actual')

        # 앙상블 모델이 2개 이상이면 개별 모델도 표시
        if num_tm > 1:
            for mi, model_preds in enumerate(next_obs_per_model):
                color = _MODEL_COLORS[mi % len(_MODEL_COLORS)]
                ax.plot(steps, model_preds[:, i], color=color,
                        linewidth=0.8, linestyle=':', alpha=0.6, label=f'Model {mi}')
            ax.plot(steps, ensemble, color='green', linewidth=1.4,
                    linestyle='--', label='Ensemble avg')
        else:
            ax.plot(steps, next_obs_per_model[0][:, i], color='darkorange',
                    linewidth=1.2, linestyle='--', label='Model pred')

        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel(name, fontsize=12, fontweight='bold')
        ax.set_title(f'Obs Prediction - {name}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.4)
        _save_fig(fig, os.path.join(out_dir, f'obs_pred_{name}.png'))

        # --- 예측 오차 ---
        err_ensemble = np.abs(ensemble - real)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='white')

        if num_tm > 1:
            for mi, model_preds in enumerate(next_obs_per_model):
                color = _MODEL_COLORS[mi % len(_MODEL_COLORS)]
                err_mi = np.abs(model_preds[:, i] - real)
                ax.plot(steps, err_mi, color=color, linewidth=0.8,
                        linestyle=':', alpha=0.6, label=f'Model {mi} err')
            ax.plot(steps, err_ensemble, color='green', linewidth=1.4,
                    linestyle='--', label=f'Ensemble err (avg={err_ensemble.mean():.4f})')
        else:
            err_m = np.abs(next_obs_per_model[0][:, i] - real)
            ax.plot(steps, err_m, color='darkorange', linewidth=1.3,
                    label=f'Model err (avg={err_m.mean():.4f})')

        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('|error|', fontsize=12, fontweight='bold')
        ax.set_title(f'Prediction Error - {name}  (ensemble avg={err_ensemble.mean():.4f})',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.4)
        _save_fig(fig, os.path.join(out_dir, f'obs_pred_err_{name}.png'))


def save_csv(data, out_dir):
    path  = os.path.join(out_dir, 'step_data.csv')
    T     = len(data['reward'])
    num_tm = len(data['next_obs_per_model'])

    ensemble_err = np.abs(data['next_obs_ensemble'] - data['next_obs_real'])

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = (
            ['step', 'reward', 'cumulative_reward', 'q1', 'q2']
            + [f'action_{n}'            for n in ACT_NAMES]
            + [f'obs_{n}'               for n in OBS_DIM_NAMES]
            + [f'next_obs_real_{n}'     for n in OBS_DIM_NAMES]
            + [f'next_obs_ensemble_{n}' for n in OBS_DIM_NAMES]
            + [f'ensemble_err_{n}'      for n in OBS_DIM_NAMES]
        )
        for mi in range(num_tm):
            header += [f'model{mi}_pred_{n}' for n in OBS_DIM_NAMES]
        writer.writerow(header)

        cumrew = np.cumsum(data['reward'])
        for t in range(T):
            row = (
                [t + 1, data['reward'][t], cumrew[t], data['q1'][t], data['q2'][t]]
                + data['action'][t].tolist()
                + data['obs'][t].tolist()
                + data['next_obs_real'][t].tolist()
                + data['next_obs_ensemble'][t].tolist()
                + ensemble_err[t].tolist()
            )
            for mi in range(num_tm):
                row += data['next_obs_per_model'][mi][t].tolist()
            writer.writerow(row)

    print(f'  Saved: {path}')


def save_video(frames, out_dir, fps=50):
    if not frames:
        print('  [video] no frames - skip')
        return
    path = os.path.join(out_dir, 'episode.mp4')
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps)
        print(f'  Saved: {path}  (imageio, {len(frames)} frames)')
        return
    except ImportError:
        pass
    try:
        import cv2
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f'  Saved: {path}  (cv2, {len(frames)} frames)')
        return
    except ImportError:
        pass
    print('  [video] imageio or opencv-python required.')
    print('    pip install imageio[ffmpeg]   or   pip install opencv-python')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='MBRL test script')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='체크포인트 파일 경로 (직접 지정)')
    p.add_argument('--checkpoint-seed', type=int, default=1,
                   help='학습 시드 번호 (MBRL_Seed{N}_BestModel.pt 로드)')
    p.add_argument('--checkpoint-dir', type=str, default=None,
                   help='체크포인트 디렉토리 (기본: ./checkpoints/mbrl_real_ratio=0.2)')
    p.add_argument('--test-seed', type=int, default=42,
                   help='테스트 재현용 시드')
    p.add_argument('--fps', type=int, default=50, help='영상 FPS')
    p.add_argument('--out-dir', type=str, default=None,
                   help='저장 디렉토리 (기본: ./figures/test_mbrl/ckpt{N}_test{M})')
    return p.parse_args()


def main():
    args = parse_args()

    # 체크포인트 경로 결정
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_dir = args.checkpoint_dir or os.path.join(
            _CYCLE_DIR, 'checkpoints', 'mbrl_real_ratio=0.2')
        ckpt_path = os.path.join(ckpt_dir, f'MBRL_Seed{args.checkpoint_seed}_BestModel.pt')

    if not os.path.exists(ckpt_path):
        print(f'[ERROR] checkpoint not found: {ckpt_path}')
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(
        _CYCLE_DIR, 'figures', 'test_mbrl',
        f'ckpt{args.checkpoint_seed}_test{args.test_seed}',
    )
    os.makedirs(out_dir, exist_ok=True)

    # 시드 고정
    np.random.seed(args.test_seed)
    torch.manual_seed(args.test_seed)

    actor, critic1, critic2, transition_models = load_checkpoint(ckpt_path)

    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    env.reset(seed=args.test_seed)

    print(f'[test_mbrl] test_seed={args.test_seed}  out_dir={out_dir}')

    data = run_episode(env, actor, critic1, critic2, transition_models)
    env.close()

    total_reward = data['reward'].sum()
    T = len(data['reward'])
    ensemble_err_avg = np.abs(data['next_obs_ensemble'] - data['next_obs_real']).mean()
    print(f'[test_mbrl] Episode done - steps={T}  total_reward={total_reward:.2f}')
    print(f'  ensemble model error avg: {ensemble_err_avg:.5f}')

    print('[test_mbrl] Saving figures...')
    save_reward(data['reward'], out_dir)
    save_q_value(data['q1'], data['q2'], out_dir)
    save_actions(data['action'], out_dir)
    save_obs_predictions(data['next_obs_real'], data['next_obs_per_model'],
                         data['next_obs_ensemble'], out_dir)
    save_csv(data, out_dir)
    save_video(data['frames'], out_dir, fps=args.fps)

    print(f'\n[test_mbrl] Done  ->  {out_dir}')
    print(f'  steps       : {T}')
    print(f'  total reward: {total_reward:.2f}')
    print(f'  files saved : reward, q_value, action x4, obs_pred x14, obs_pred_err x14, CSV, MP4')


if __name__ == '__main__':
    main()
