"""
main.py — 단일 시드 학습 진입점
Usage:
  python main.py --checkpoint-dir ./checkpoints --seed 0
  python main.py --checkpoint-dir ./checkpoints --seed 0 --resume
"""
import argparse
import csv
import os
import pickle

from dob_mbrl.training import train_DOB_core, DOBMBRLConfig


def parse_args():
    parser = argparse.ArgumentParser(description='DOB-MBRL single-seed training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/baseline',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=1,
                        help='랜덤 시드 (1-based)')
    parser.add_argument('--resume', action='store_true',
                        help='마지막 체크포인트에서 재개')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = DOBMBRLConfig()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(os.path.dirname(args.checkpoint_dir), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    print(f'[main] seed={args.seed}  checkpoint_dir={args.checkpoint_dir}  resume={args.resume}')

    rewards, steps = train_DOB_core(
        run_idx        = args.seed,
        num_episodes   = cfg.num_episodes,
        result_queue   = None,
        checkpoint_dir = args.checkpoint_dir,
        resume         = args.resume,
        cfg            = cfg,
    )

    log_path = os.path.join(log_dir, f'seed_{args.seed}_result.pkl')
    with open(log_path, 'wb') as f:
        pickle.dump({'rewards': rewards, 'steps': steps}, f)
    print(f'[main] Saved log → {log_path}')

    csv_path = os.path.join(log_dir, f'seed_{args.seed}_result.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_steps', 'reward'])
        for ep, (s, r) in enumerate(zip(steps, rewards), start=1):
            writer.writerow([ep, s, r])
    print(f'[main] Saved log → {csv_path}')


if __name__ == '__main__':
    main()
