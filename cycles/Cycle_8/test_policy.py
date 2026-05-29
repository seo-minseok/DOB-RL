"""
test_policy.py — 저장된 체크포인트로 결정론적 정책 테스트 + MP4 저장 (Hopper-v5)
Usage:
  python test_policy.py --seed 1                  # checkpoints/Champion_Seed1_BestModel.pt
  python test_policy.py --seed 1 --num-episodes 3
"""
import argparse
import os
import sys

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dob_mbrl.models import ActorNetwork


def run_episode(actor, seed: int):
    """노이즈 없는 결정론적 정책으로 한 에피소드 실행. (frames, reward, steps) 반환."""
    import gymnasium as gym

    render_env = gym.make('Hopper-v5', render_mode='rgb_array')
    obs_raw, _ = render_env.reset(seed=seed)
    obs = np.array(obs_raw, dtype=np.float32)

    frames = []
    episode_reward = 0.0

    for step in range(999):
        frames.append(render_env.render())

        obs_t = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = actor(obs_t).cpu().numpy().flatten()
        action = np.clip(action, -1.0, 1.0)

        result = render_env.step(action)
        next_obs, reward, terminated, truncated, _ = result
        done = terminated or truncated

        episode_reward += reward
        obs = np.array(next_obs, dtype=np.float32)

        if done:
            frames.append(render_env.render())
            break

    render_env.close()
    return frames, episode_reward, step + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='학습 시드 (체크포인트 선택)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--output-dir', type=str, default='./videos')
    parser.add_argument('--num-episodes', type=int, default=3, help='녹화할 에피소드 수')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.checkpoint_dir, f'Champion_Seed{args.seed}_BestModel.pt')
    if not os.path.exists(ckpt_path):
        print(f'[ERROR] 체크포인트 없음: {ckpt_path}')
        print('  → 해당 시드가 TARGET_SCORE에 도달한 적이 없으면 저장되지 않습니다.')
        return

    ckpt = torch.load(ckpt_path, weights_only=False)
    actor = ActorNetwork(11, 3)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    print(f'[Seed {args.seed}] Checkpoint loaded  (saved at episode {ckpt["episode"]}, '
          f'total_steps={ckpt["total_steps"]})')

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(args.num_episodes):
        frames, ep_reward, ep_steps = run_episode(actor, seed=ep)
        out_path = os.path.join(args.output_dir, f'seed{args.seed}_ep{ep + 1}.mp4')
        imageio.mimwrite(out_path, frames, fps=args.fps)
        print(f'  Episode {ep + 1}: reward={ep_reward:.2f}  steps={ep_steps}  → {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
