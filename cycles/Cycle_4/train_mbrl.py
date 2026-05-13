"""
train_mbrl.py — Pure MBRL single-seed 학습 진입점
BipedalWalker + TD3 + Ensemble Transition Models (DOB 없음, train_MBRL_core.py 방식)

Usage:
  python train_mbrl.py --seed 1 --real-ratio 0.5
  python train_mbrl.py --seed 1 --real-ratio 1.0          # vanilla TD3 (모델 rollout 비활성)
  python train_mbrl.py --seed 1 --real-ratio 0.5 --resume
"""
import argparse
import dataclasses
import json
import os

from dob_mbrl.training.trainer_mbrl import train_MBRL_core
from dob_mbrl.training.config import DOBMBRLConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Pure MBRL training (BipedalWalker TD3)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_mbrl',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=1,
                        help='랜덤 시드 (1-based)')
    parser.add_argument('--resume', action='store_true',
                        help='마지막 체크포인트에서 재개')
    parser.add_argument('--real-ratio', type=float, default=None,
                        help='real 데이터 비율 (0~1, 미입력 시 config.py 값 사용)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = DOBMBRLConfig()

    if args.real_ratio is not None:
        cfg.real_ratio = args.real_ratio

    run_name = f'mbrl_real_ratio={cfg.real_ratio}'
    _here       = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_here, 'results', run_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    config_path = os.path.join(results_dir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)
        print(f'[train_mbrl] config saved → {config_path}')

    print(f'[train_mbrl] seed={args.seed}  real_ratio={cfg.real_ratio}  '
          f'checkpoint_dir={args.checkpoint_dir}  resume={args.resume}')
    print(f'[train_mbrl] results_dir={results_dir}')

    train_MBRL_core(
        run_idx        = args.seed,
        num_episodes   = cfg.num_episodes,
        checkpoint_dir = args.checkpoint_dir,
        resume         = args.resume,
        cfg            = cfg,
        results_dir    = results_dir,
    )


if __name__ == '__main__':
    main()
