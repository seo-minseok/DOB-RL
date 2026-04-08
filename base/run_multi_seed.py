"""
run_multi_seed.py — 멀티 시드 병렬 학습 진입점
MATLAB: Multi_Seed_DOB_Exp.m
Usage:
  python run_multi_seed.py --checkpoint-dir ./checkpoints --num-seeds 16
"""
import argparse
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

    # --- Interpolation onto common step axis ---
    max_total_step = max(max(steps) for steps in all_steps if steps)
    common_steps   = np.linspace(0, max_total_step, 1000)
    interp_rewards = np.full((num_runs, 1000), np.nan, dtype=np.float64)

    for i in range(num_runs):
        x_run = np.array([0] + list(all_steps[i]),   dtype=np.float64)
        y_run = np.array([0] + list(all_rewards[i]), dtype=np.float64)

        _, unique_idx = np.unique(x_run, return_index=True)
        x_run = x_run[unique_idx]
        y_run = y_run[unique_idx]

        if len(x_run) < 2:
            continue

        vals = np.interp(common_steps, x_run, y_run, left=np.nan, right=np.nan)
        last_step = x_run[-1]
        last_val  = y_run[-1]
        vals[common_steps > last_step] = last_val
        interp_rewards[i, :] = vals

    mean_curve = np.nanmean(interp_rewards, axis=0)
    std_curve  = np.nanstd( interp_rewards, axis=0)

    # 요약 출력
    print(f'Final mean reward (last 10 ep avg across seeds): '
          f'{np.nanmean([np.mean(r[-10:]) for r in all_rewards if r]):.1f}')
    print(f'Target score: {target_score}')
    print(f'\nRun: python scripts/plot_results.py --results-dir {results_dir} '
          f'to generate figures.')


if __name__ == '__main__':
    main()
