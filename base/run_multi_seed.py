"""
run_multi_seed.py — 멀티 시드 병렬 학습 진입점
MATLAB: Multi_Seed_DOB_Exp.m
Usage:
  python run_multi_seed.py --checkpoint-dir ./checkpoints --num-seeds 16
"""
import argparse
import csv
import multiprocessing
import os
import pickle
import time

import numpy as np

from dob_mbrl.training import train_DOB_core, DOBMBRLConfig


def _run_single(args):
    run_idx, num_episodes, checkpoint_dir = args
    print(f'[Run {run_idx:02d}] DOB-MBRL training started...')
    cfg = DOBMBRLConfig()
    rewards, steps = train_DOB_core(
        run_idx        = run_idx,
        num_episodes   = num_episodes,
        checkpoint_dir = checkpoint_dir,
        cfg            = cfg,
    )
    print(f'[Run {run_idx:02d}] Done.')
    return rewards, steps


def parse_args():
    parser = argparse.ArgumentParser(description='DOB-MBRL multi-seed training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--num-seeds', type=int, default=16,
                        help='병렬 실행 시드 수')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = DOBMBRLConfig()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    results_dir = os.path.join(os.path.dirname(args.checkpoint_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)

    num_runs     = args.num_seeds
    max_episodes = cfg.num_episodes
    target_score = 480.0

    print('=' * 60)
    print(f'DOB-MBRL Multi-Seed Training  (runs={num_runs}, ep={max_episodes})')
    print('=' * 60)
    start_time = time.time()

    num_workers = min(num_runs, multiprocessing.cpu_count())
    args_list   = [(i + 1, max_episodes, args.checkpoint_dir) for i in range(num_runs)]

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_run_single, args_list)
    except (OSError, PermissionError) as exc:
        print(f'Parallel unavailable ({exc}). Falling back to sequential.')
        results = [_run_single(a) for a in args_list]

    all_rewards = [r[0] for r in results]
    all_steps   = [r[1] for r in results]

    total_time = time.time() - start_time
    print(f'\n=== All training complete ({total_time / 60:.1f} min) ===')

    result_path = os.path.join(results_dir, 'DOB_MBRL_MultiSeed_Result.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump({'all_rewards': all_rewards, 'all_steps': all_steps}, f)
    print(f'Saved: {result_path}')

    csv_path = os.path.join(results_dir, 'DOB_MBRL_MultiSeed_Result.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'episode', 'total_steps', 'reward'])
        for seed_idx, (rewards_s, steps_s) in enumerate(zip(all_rewards, all_steps), start=1):
            for ep, (s, r) in enumerate(zip(steps_s, rewards_s), start=1):
                writer.writerow([seed_idx, ep, s, r])
    print(f'Saved: {csv_path}')

    # 요약 출력
    print(f'Final mean reward (last 10 ep avg across seeds): '
          f'{np.nanmean([np.mean(r[-10:]) for r in all_rewards if r]):.1f}')
    print(f'Target score: {target_score}')
    print(f'\nRun: python scripts/plot_results.py --results-dir {results_dir} '
          f'to generate figures.')


if __name__ == '__main__':
    main()
