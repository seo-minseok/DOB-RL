"""
render_test.py — Cycle 5 체크포인트 테스트 & 영상 녹화

Usage:
  cd cycles/Cycle_5
  python scripts/render_test.py --run real_ratio=0.2_uncert_thresh=0.4 --seed 1
  python scripts/render_test.py --run real_ratio=0.2_uncert_thresh=0.4 --seed 1 --episodes 3
"""
import argparse
import os
import sys

import numpy as np
import torch

# cycles/Cycle_5 를 sys.path에 추가 (scripts/ 하위에서 실행 시)
_CYCLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CYCLE_DIR not in sys.path:
    sys.path.insert(0, _CYCLE_DIR)

from dob_mbrl.models import ActorNetwork
from dob_mbrl.dynamics import ACT_DIM

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

# 체크포인트에서 obs_dim을 읽어 알맞은 obs 필터를 결정
# raw 24D BipedalWalker obs 기준:
#   lidar 제거(14-23): 14D  → 인덱스 [0..13]
#   contact 추가 제거(8,13): 12D → 인덱스 [0,1,2,3,4,5,6,7,9,10,11,12]
_OBS_KEEP_14 = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=np.int32)
_OBS_KEEP_12 = np.array([0,1,2,3,4,5,6,7,9,10,11,12],     dtype=np.int32)


def _get_obs_keep(obs_dim: int) -> np.ndarray:
    if obs_dim == 14:
        return _OBS_KEEP_14
    elif obs_dim == 12:
        return _OBS_KEEP_12
    raise ValueError(f"알 수 없는 obs_dim: {obs_dim}")


def _reset(env, obs_keep) -> np.ndarray:
    result = env.reset()
    raw = result[0] if isinstance(result, tuple) else result
    return np.asarray(raw, dtype=np.float32)[obs_keep]


def _step(env, action, obs_keep) -> tuple:
    result = env.step(action)
    if len(result) == 5:
        raw, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        raw, reward, done, info = result
        done = bool(done)
    return np.asarray(raw, dtype=np.float32)[obs_keep], reward, done, info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=str, default='real_ratio=0.2_uncert_thresh=0.4',
                   help='체크포인트 서브폴더 이름')
    p.add_argument('--seed', type=int, default=1, help='시드 번호')
    p.add_argument('--episodes', type=int, default=3, help='녹화할 에피소드 수')
    p.add_argument('--fps', type=int, default=50, help='영상 FPS')
    p.add_argument('--out-dir', type=str, default=None,
                   help='영상 저장 폴더 (기본: cycles/Cycle_5/videos/<run>/)')
    return p.parse_args()


def load_actor(checkpoint_path: str):
    """체크포인트에서 obs_dim을 자동 검출 후 ActorNetwork 로드."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    actor_sd = ckpt['actor']
    obs_dim = actor_sd['obs_min'].shape[0]
    actor = ActorNetwork(obs_dim, ACT_DIM)
    # register_buffer가 전역 OBS_MIN/OBS_MAX(14D)로 초기화되므로 체크포인트 값으로 교체
    actor.register_buffer('obs_min', actor_sd['obs_min'].clone())
    actor.register_buffer('obs_max', actor_sd['obs_max'].clone())
    actor.load_state_dict(actor_sd)
    actor.eval()
    print(f"  [load_actor] obs_dim={obs_dim}  act_dim={ACT_DIM}")
    return actor, obs_dim


@torch.no_grad()
def select_action(actor: ActorNetwork, obs: np.ndarray) -> np.ndarray:
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action = actor(obs_t).squeeze(0).numpy()
    return action


def run_episode(env, actor: ActorNetwork, obs_keep: np.ndarray):
    """단일 에피소드 실행. Returns (total_reward, frames list)."""
    obs = _reset(env, obs_keep)
    frames = []
    total_reward = 0.0
    done = False

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = select_action(actor, obs)
        obs, reward, done, _ = _step(env, action, obs_keep)
        total_reward += reward

    return total_reward, frames


def save_video(frames: list, path: str, fps: int):
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps)
        return 'imageio'
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
        return 'cv2'
    except ImportError:
        pass

    raise RuntimeError(
        "imageio 또는 opencv-python 이 필요합니다.\n"
        "  pip install imageio[ffmpeg]   또는\n"
        "  pip install opencv-python"
    )


def main():
    args = parse_args()

    ckpt_path = os.path.join(
        _CYCLE_DIR, 'checkpoints', args.run,
        f'Champion_Seed{args.seed}_BestModel.pt'
    )
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] 체크포인트를 찾을 수 없습니다: {ckpt_path}")
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(_CYCLE_DIR, 'videos', args.run)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[render_test] 체크포인트 로드: {ckpt_path}")
    actor, obs_dim = load_actor(ckpt_path)
    obs_keep = _get_obs_keep(obs_dim)

    env = gym.make('BipedalWalker-v3', render_mode='rgb_array')

    all_rewards = []
    for ep in range(1, args.episodes + 1):
        total_reward, frames = run_episode(env, actor, obs_keep)
        all_rewards.append(total_reward)
        print(f"  Episode {ep}: reward={total_reward:.1f}  frames={len(frames)}")

        video_path = os.path.join(out_dir, f'seed{args.seed}_ep{ep}.mp4')
        backend = save_video(frames, video_path, args.fps)
        print(f"  -> 저장: {video_path}  (backend={backend})")

    env.close()

    print(f"\n[render_test] 완료")
    print(f"  에피소드 수  : {args.episodes}")
    print(f"  평균 reward  : {np.mean(all_rewards):.1f}")
    print(f"  최고 reward  : {np.max(all_rewards):.1f}")
    print(f"  영상 저장 위치: {out_dir}")


if __name__ == '__main__':
    main()
