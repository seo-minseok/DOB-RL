"""
run_multi_seed_mbrl.py — Pure MBRL 멀티 시드 병렬 학습 진입점
BipedalWalker + TD3 + Ensemble Transition Models (DOB 없음)

Usage:
  python run_multi_seed_mbrl.py --num-seeds 16 --real-ratio 0.5
  python run_multi_seed_mbrl.py --num-seeds 16 --real-ratio 1.0
"""
import argparse
import dataclasses
import json
import multiprocessing
import os
import time

import numpy as np

from dob_mbrl.training.trainer_mbrl import train_MBRL_core
from dob_mbrl.training.config import DOBMBRLConfig


def _run_single(args):
    run_idx, num_episodes, checkpoint_dir, results_dir, real_ratio = args
    print(f'[MBRL Run {run_idx:02d}] started...')
    cfg = DOBMBRLConfig()
    cfg.real_ratio = real_ratio
    rewards, steps = train_MBRL_core(
        run_idx        = run_idx,
        num_episodes   = num_episodes,
        checkpoint_dir = checkpoint_dir,
        cfg            = cfg,
        results_dir    = results_dir,
    )
    print(f'[MBRL Run {run_idx:02d}] Done. Final reward: {np.mean(rewards[-10:]):.1f}')
    return rewards, steps


def parse_args():
    parser = argparse.ArgumentParser(description='Pure MBRL multi-seed training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_mbrl',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--num-seeds', type=int, default=16,
                        help='병렬 실행 시드 수')
    parser.add_argument('--real-ratio', type=float, default=None,
                        help='real 데이터 비율 (미입력 시 config.py 값 사용)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = DOBMBRLConfig()
    if args.real_ratio is not None:
        cfg.real_ratio = args.real_ratio

    run_name    = f'mbrl_real_ratio={cfg.real_ratio}'
    _here       = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_here, 'results', run_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # config.json 저장 (최초 1회)
    config_path = os.path.join(results_dir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)
        print(f'[run_multi_seed_mbrl] config saved → {config_path}')

    num_runs = args.num_seeds
    print('=' * 60)
    print(f'Pure MBRL BipedalWalker (TD3)  runs={num_runs}  ep={cfg.num_episodes}  real_ratio={cfg.real_ratio}')
    print('=' * 60)

    start_time  = time.time()
    num_workers = min(num_runs, multiprocessing.cpu_count())
    args_list   = [
        (i + 1, cfg.num_episodes, args.checkpoint_dir, results_dir, cfg.real_ratio)
        for i in range(num_runs)
    ]

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_run_single, args_list)
    except (OSError, PermissionError) as exc:
        print(f'Parallel unavailable ({exc}). Falling back to sequential.')
        results = [_run_single(a) for a in args_list]

    total_time  = time.time() - start_time
    all_rewards = [r[0] for r in results]

    print(f'\n=== All training complete ({total_time / 60:.1f} min) ===')
    print(f'Final mean reward (last 10 ep avg across seeds): '
          f'{np.nanmean([np.mean(r[-10:]) for r in all_rewards if r]):.1f}')
    print(f'\nCSV files saved in: {results_dir}')
    print(f'  seed_1_progress.csv ~ seed_{num_runs}_progress.csv')


if __name__ == '__main__':
    main()
