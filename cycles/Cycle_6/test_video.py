"""
test_video.py — 저장된 체크포인트로 BipedalWalker 테스트 후 영상 저장
Usage:
  python test_video.py --checkpoint ./checkpoints_real_ratio=0.2_uncert_thresh=0.4/Champion_Seed1_BestModel.pt
  python test_video.py --checkpoint ./checkpoints_real_ratio=0.2_uncert_thresh=0.4/Champion_Seed1_BestModel.pt --num-episodes 3
"""
import argparse
import os
import sys
import numpy as np
import torch
import imageio

# Cycle_5 루트를 sys.path에 추가
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dob_mbrl.models import ActorNetwork
from dob_mbrl.dynamics.constants import OBS_DIM, ACT_DIM
from dob_mbrl.envs.bipedalwalker_utils import step_env

_OBS_KEEP = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)


def make_render_env():
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        import gym
    return gym.make('BipedalWalker-v3', render_mode='rgb_array')


def reset_render_env(env):
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    return np.asarray(obs, dtype=np.float32)[_OBS_KEEP]


def run_episode(env, actor, device):
    obs = reset_render_env(env)
    frames = []
    total_reward = 0.0
    step = 0
    done = False

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame.astype(np.uint8))

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = actor(obs_t).squeeze(0).cpu().numpy()

        obs, reward, done, _ = step_env(env, action)
        total_reward += reward
        step += 1

        if step >= 1600:
            break

    return frames, total_reward, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-episodes', type=int, default=2)
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f'[ERROR] checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    ckpt_dir  = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir   = args.output_dir or os.path.join(ckpt_dir, 'videos')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cpu')
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=False)

    episode_saved = ckpt.get('episode', '?')
    total_steps   = ckpt.get('total_steps', '?')
    print(f'[test_video] checkpoint: episode={episode_saved}, total_steps={total_steps}')

    actor = ActorNetwork(OBS_DIM, ACT_DIM).to(device)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    env = make_render_env()

    for ep in range(1, args.num_episodes + 1):
        print(f'[test_video] Episode {ep}/{args.num_episodes} ...', end=' ', flush=True)
        frames, total_reward, steps = run_episode(env, actor, device)
        print(f'reward={total_reward:.1f}  steps={steps}  frames={len(frames)}')

        out_path = os.path.join(out_dir, f'{ckpt_name}_ep{ep}.mp4')
        if frames:
            imageio.mimwrite(out_path, frames, fps=args.fps, macro_block_size=1)
            print(f'  → saved: {out_path}')
        else:
            print(f'  [WARN] no frames captured for episode {ep}')

    env.close()
    print('[test_video] done.')


if __name__ == '__main__':
    main()
