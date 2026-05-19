"""
main_mbrl.py — Pure MBRL (TD3 + ensemble transition) 단일 시드 학습 진입점
Usage:
  python main_mbrl.py --seed 1
  python main_mbrl.py --seed 1 --real-ratio 0.2
  python main_mbrl.py --seed 1 --resume
"""
import argparse
import dataclasses
import json
import os

import torch

from dob_mbrl.training import train_MBRL_core, DOBMBRLConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Pure MBRL single-seed training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--real-ratio', type=float, default=None)
    parser.add_argument('--num-threads', type=int, default=None,
                        help='PyTorch intra-op 스레드 수 (병렬 실행 시 CPU 분배용, 예: --num-threads 8)')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(args.num_threads)
        print(f'[MBRL] num_threads={args.num_threads}')
    cfg  = DOBMBRLConfig()

    if args.real_ratio is not None:
        cfg.real_ratio = args.real_ratio

    run_name    = f'mbrl_real_ratio={cfg.real_ratio}'
    _here       = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_here, 'results', run_name)
    os.makedirs(results_dir, exist_ok=True)

    config_path = os.path.join(results_dir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)

    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f'[MBRL] seed={args.seed}  real_ratio={cfg.real_ratio}  checkpoint_dir={checkpoint_dir}')
    print(f'[MBRL] results_dir={results_dir}')

    train_MBRL_core(
        run_idx        = args.seed,
        num_episodes   = cfg.num_episodes,
        checkpoint_dir = checkpoint_dir,
        resume         = args.resume,
        cfg            = cfg,
        results_dir    = results_dir,
    )


if __name__ == '__main__':
    main()
