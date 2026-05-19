"""
test_dob.py — DOB-MBRL 학습된 모델 테스트 및 분석

저장 파일 (figures/test/ckpt{ckpt_seed}_test{test_seed}/ 하위):
  reward.png            — 스텝별 누적 reward
  q_value.png           — 스텝별 Q1, Q2 값
  action_{name}.png     — 스텝별 행동 (4개)
  obs_pred_{name}.png   — 차원별 actual / nominal / model 예측 비교 (14개)
  obs_pred_err_{name}.png — 차원별 예측 오차 (nominal error / model error)
  step_data.csv         — 모든 스텝 데이터
  episode.mp4           — 에피소드 영상

Usage:
  python scripts/test_dob.py --checkpoint-seed 1 --test-seed 42
  python scripts/test_dob.py --checkpoint-seed 2 --test-seed 0
  python scripts/test_dob.py --checkpoint ./checkpoints/Champion_Seed1_BestModel.pt --test-seed 42
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

from dob_mbrl.models import ActorNetwork, QNetwork, ResidualDxNet, NormalizedRBFModel, ContactNet
from dob_mbrl.training.config import DOBMBRLConfig
from dob_mbrl.dynamics import (
    default_bipedalwalker_params, step_nominal_bipedalwalker,
    OBS_DIM, ACT_DIM, OBS_DIM_NAMES,
)
from dob_mbrl.dynamics.dob import predict_next_obs_dob
from dob_mbrl.envs.bipedalwalker_utils import reset_env, step_env

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

ACT_NAMES = ['hip1_motor', 'knee1_motor', 'hip2_motor', 'knee2_motor']


# ---------------------------------------------------------------------------
# Checkpoint 로드
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str):
    cfg = DOBMBRLConfig()
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

    res_net = ResidualDxNet(OBS_DIM, ACT_DIM, hidden=64)
    res_net.load_state_dict(ckpt['res_net'])
    res_net.eval()

    uncert_model = NormalizedRBFModel(cfg.num_rbf_centers, cfg.rbf_width, cfg.rbf_initial_value)
    uncert_model.load_state_dict(ckpt['uncert_model'])
    uncert_model.eval()

    contact_net = None
    if 'contact_net' in ckpt:
        contact_net = ContactNet(OBS_DIM, ACT_DIM, hidden=64)
        contact_net.load_state_dict(ckpt['contact_net'])
        contact_net.eval()

    print(f'[test_dob] Loaded checkpoint: {ckpt_path}')
    print(f'  episode={ckpt.get("episode", "?")}  total_steps={ckpt.get("total_steps", "?")}')
    return actor, critic1, critic2, res_net, uncert_model, contact_net


# ---------------------------------------------------------------------------
# 에피소드 실행
# ---------------------------------------------------------------------------

def run_episode(env, actor, critic1, critic2, res_net, uncert_model,
                contact_net, p_nom):
    """
    한 에피소드를 실행하면서 모든 분석 데이터를 수집한다.

    Returns dict with keys:
      obs, action, next_obs_real,
      next_obs_nom, next_obs_model,
      reward, q1, q2, uncert,
      done_step (에피소드 종료 스텝, 0-based)
    """
    obs = reset_env(env)
    frames = []

    obs_list        = []
    act_list        = []
    next_obs_real   = []
    next_obs_nom    = []
    next_obs_model  = []
    reward_list     = []
    q1_list         = []
    q2_list         = []
    uncert_list     = []

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

        # 관측 예측
        obs_2d = obs[np.newaxis]
        act_2d = action[np.newaxis]

        nom_next  = step_nominal_bipedalwalker(obs_2d, act_2d, p_nom)[0]
        model_next = predict_next_obs_dob(obs_2d, act_2d, res_net, p_nom,
                                          use_nominal=True,
                                          contact_net=contact_net)[0]

        # RBF uncertainty
        inp_rbf = torch.tensor(np.concatenate([obs_2d, act_2d], axis=-1))
        with torch.no_grad():
            uncert = uncert_model(inp_rbf).cpu().numpy().flatten()

        next_obs, reward, done, _ = step_env(env, action)

        obs_list.append(obs.copy())
        act_list.append(action.copy())
        next_obs_real.append(next_obs.copy())
        next_obs_nom.append(nom_next.copy())
        next_obs_model.append(model_next.copy())
        reward_list.append(float(reward))
        q1_list.append(float(q1_val))
        q2_list.append(float(q2_val))
        uncert_list.append(np.linalg.norm(uncert))

        obs = next_obs

    return {
        'obs':            np.array(obs_list),
        'action':         np.array(act_list),
        'next_obs_real':  np.array(next_obs_real),
        'next_obs_nom':   np.array(next_obs_nom),
        'next_obs_model': np.array(next_obs_model),
        'reward':         np.array(reward_list),
        'q1':             np.array(q1_list),
        'q2':             np.array(q2_list),
        'uncert':         np.array(uncert_list),
        'frames':         frames,
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
    steps = np.arange(1, len(reward) + 1)
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


def save_uncert(uncert, out_dir):
    steps = np.arange(1, len(uncert) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='white')
    ax.plot(steps, uncert, color='purple', linewidth=1.5)
    ax.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('||uncertainty||₂', fontsize=12, fontweight='bold')
    ax.set_title('RBF Uncertainty (||pred||₂)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.4)
    _save_fig(fig, os.path.join(out_dir, 'uncertainty.png'))


def save_actions(action, out_dir):
    steps = np.arange(1, len(action) + 1)
    for i, name in enumerate(ACT_NAMES):
        fig, ax = plt.subplots(1, 1, figsize=(10, 3), facecolor='white')
        ax.plot(steps, action[:, i], color='steelblue', linewidth=1.3)
        ax.axhline( 1.0, color='red',  linewidth=0.8, linestyle='--', alpha=0.6)
        ax.axhline(-1.0, color='red',  linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Action', fontsize=12, fontweight='bold')
        ax.set_title(f'Action — {name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.4)
        _save_fig(fig, os.path.join(out_dir, f'action_{name}.png'))


def save_obs_predictions(next_obs_real, next_obs_nom, next_obs_model, out_dir):
    steps = np.arange(1, len(next_obs_real) + 1)
    for i, name in enumerate(OBS_DIM_NAMES):
        real  = next_obs_real[:, i]
        nom   = next_obs_nom[:, i]
        model = next_obs_model[:, i]

        # --- 예측 vs 실제 ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='white')
        ax.plot(steps, real,  color='steelblue',  linewidth=1.8, label='Actual')
        ax.plot(steps, nom,   color='darkorange',  linewidth=1.2, linestyle='--', label='Nominal', alpha=0.8)
        ax.plot(steps, model, color='green',       linewidth=1.2, linestyle='--', label='Nom+Residual', alpha=0.8)
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel(name, fontsize=12, fontweight='bold')
        ax.set_title(f'Obs Prediction — {name}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.4)
        _save_fig(fig, os.path.join(out_dir, f'obs_pred_{name}.png'))

        # --- 예측 오차 ---
        err_nom   = np.abs(nom   - real)
        err_model = np.abs(model - real)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), facecolor='white')
        ax.plot(steps, err_nom,   color='darkorange', linewidth=1.2, label='Nominal error', alpha=0.8)
        ax.plot(steps, err_model, color='green',      linewidth=1.2, label='Model error',   alpha=0.8)
        ax.set_xlabel('Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('|error|', fontsize=12, fontweight='bold')
        ax.set_title(f'Prediction Error — {name}  '
                     f'(nom={err_nom.mean():.4f}, model={err_model.mean():.4f})',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.4)
        _save_fig(fig, os.path.join(out_dir, f'obs_pred_err_{name}.png'))


def save_csv(data, out_dir):
    path = os.path.join(out_dir, 'step_data.csv')
    T = len(data['reward'])
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = (
            ['step', 'reward', 'cumulative_reward', 'q1', 'q2', 'uncert_norm']
            + [f'action_{n}'  for n in ACT_NAMES]
            + [f'obs_{n}'     for n in OBS_DIM_NAMES]
            + [f'next_obs_real_{n}'  for n in OBS_DIM_NAMES]
            + [f'next_obs_nom_{n}'   for n in OBS_DIM_NAMES]
            + [f'next_obs_model_{n}' for n in OBS_DIM_NAMES]
            + [f'nom_err_{n}'   for n in OBS_DIM_NAMES]
            + [f'model_err_{n}' for n in OBS_DIM_NAMES]
        )
        writer.writerow(header)

        cumrew = np.cumsum(data['reward'])
        nom_err   = np.abs(data['next_obs_nom']   - data['next_obs_real'])
        model_err = np.abs(data['next_obs_model'] - data['next_obs_real'])

        for t in range(T):
            row = (
                [t + 1, data['reward'][t], cumrew[t], data['q1'][t], data['q2'][t], data['uncert'][t]]
                + data['action'][t].tolist()
                + data['obs'][t].tolist()
                + data['next_obs_real'][t].tolist()
                + data['next_obs_nom'][t].tolist()
                + data['next_obs_model'][t].tolist()
                + nom_err[t].tolist()
                + model_err[t].tolist()
            )
            writer.writerow(row)
    print(f'  Saved: {path}')


def save_video(frames, out_dir, fps=50):
    if not frames:
        print('  [video] 프레임 없음 — 스킵')
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
    print('  [video] imageio 또는 opencv-python 이 필요합니다.')
    print('    pip install imageio[ffmpeg]   또는   pip install opencv-python')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='DOB-MBRL test script')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='체크포인트 파일 경로 (직접 지정)')
    p.add_argument('--checkpoint-seed', type=int, default=1,
                   help='학습 시드 번호 (Champion_Seed{N}_BestModel.pt 로드)')
    p.add_argument('--checkpoint-dir', type=str, default=None,
                   help='체크포인트 디렉토리 (기본: ./checkpoints)')
    p.add_argument('--test-seed', type=int, default=42,
                   help='테스트 재현용 시드 (환경 및 torch 난수 고정)')
    p.add_argument('--fps', type=int, default=50, help='영상 FPS')
    p.add_argument('--out-dir', type=str, default=None,
                   help='저장 디렉토리 (기본: ./figures/test/ckpt{N}_test{M})')
    return p.parse_args()


def main():
    args = parse_args()

    # 체크포인트 경로 결정
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_dir = args.checkpoint_dir or os.path.join(_CYCLE_DIR, 'checkpoints')
        ckpt_path = os.path.join(ckpt_dir, f'Champion_Seed{args.checkpoint_seed}_BestModel.pt')

    if not os.path.exists(ckpt_path):
        print(f'[ERROR] 체크포인트를 찾을 수 없습니다: {ckpt_path}')
        sys.exit(1)

    # 출력 디렉토리
    out_dir = args.out_dir or os.path.join(
        _CYCLE_DIR, 'figures', 'test',
        f'ckpt{args.checkpoint_seed}_test{args.test_seed}',
    )
    os.makedirs(out_dir, exist_ok=True)

    # 시드 고정
    np.random.seed(args.test_seed)
    torch.manual_seed(args.test_seed)

    # 모델 로드
    actor, critic1, critic2, res_net, uncert_model, contact_net = load_checkpoint(ckpt_path)
    p_nom = default_bipedalwalker_params()

    # 환경 생성 (rgb_array 렌더링)
    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    env.reset(seed=args.test_seed)

    print(f'[test_dob] test_seed={args.test_seed}  out_dir={out_dir}')

    # 에피소드 실행
    data = run_episode(env, actor, critic1, critic2, res_net, uncert_model,
                       contact_net, p_nom)
    env.close()

    total_reward = data['reward'].sum()
    T = len(data['reward'])
    print(f'[test_dob] Episode done - steps={T}  total_reward={total_reward:.2f}')

    nom_err_avg   = np.abs(data['next_obs_nom']   - data['next_obs_real']).mean()
    model_err_avg = np.abs(data['next_obs_model'] - data['next_obs_real']).mean()
    print(f'  nominal error avg : {nom_err_avg:.5f}')
    print(f'  model   error avg : {model_err_avg:.5f}')

    # 저장
    print('[test_dob] Saving figures...')
    save_reward(data['reward'], out_dir)
    save_q_value(data['q1'], data['q2'], out_dir)
    save_uncert(data['uncert'], out_dir)
    save_actions(data['action'], out_dir)
    save_obs_predictions(data['next_obs_real'], data['next_obs_nom'],
                         data['next_obs_model'], out_dir)
    save_csv(data, out_dir)
    save_video(data['frames'], out_dir, fps=args.fps)

    print(f'\n[test_dob] Done  ->  {out_dir}')
    print(f'  steps       : {T}')
    print(f'  total reward: {total_reward:.2f}')
    print(f'  files saved : reward, q_value, uncertainty, action x4, obs_pred x14, obs_pred_err x14, CSV, MP4')


if __name__ == '__main__':
    main()
