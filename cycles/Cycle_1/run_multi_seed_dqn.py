"""
run_multi_seed_dqn.py — DQN 멀티 시드 병렬 학습 진입점
MATLAB: Multi_Seed_DQN_Exp.m
Usage:
  python run_multi_seed_dqn.py [--num-seeds 16]
"""
import argparse
import csv
import multiprocessing
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Cycle 1 로컬 train_DQN_core 사용 (td_loss 반환 버전)
_CYCLE_DIR = str(Path(__file__).resolve().parent)
if _CYCLE_DIR not in sys.path:
    sys.path.insert(0, _CYCLE_DIR)


def _run_single(args):
    run_idx, num_episodes, dqn_ckpt_dir = args

    # train_DQN_core의 RESULTS_DIR은 모듈 로드 시점에 결정되므로
    # 환경변수를 먼저 설정한 뒤 임포트한다 (자식 프로세스에서 fresh import)
    os.environ['DQN_RESULTS_DIR'] = dqn_ckpt_dir
    from train_DQN_core import train_DQN_core

    print(f'[Run {run_idx:02d}] DQN training started...')
    rewards, steps, td_losses = train_DQN_core(run_idx, num_episodes)
    print(f'[Run {run_idx:02d}] Done.')
    return rewards, steps, td_losses


def parse_args():
    parser = argparse.ArgumentParser(description='DQN multi-seed training')
    parser.add_argument('--num-seeds', type=int, default=16,
                        help='병렬 실행 시드 수')
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir      = os.path.dirname(os.path.abspath(__file__))
    results_dir     = os.path.join(script_dir, 'results')
    dqn_ckpt_dir    = os.path.join(script_dir, 'checkpoints', 'dqn')
    os.makedirs(results_dir,  exist_ok=True)
    os.makedirs(dqn_ckpt_dir, exist_ok=True)

    num_runs     = args.num_seeds
    max_episodes = 200
    target_score = 480.0

    print('=' * 60)
    print(f'DQN Multi-Seed Training  (runs={num_runs}, ep={max_episodes})')
    print('=' * 60)
    start_time = time.time()

    num_workers = min(num_runs, multiprocessing.cpu_count())
    args_list   = [(i + 1, max_episodes, dqn_ckpt_dir) for i in range(num_runs)]

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_run_single, args_list)
    except (OSError, PermissionError) as exc:
        print(f'Parallel unavailable ({exc}). Falling back to sequential.')
        results = [_run_single(a) for a in args_list]

    all_rewards   = [r[0] for r in results]
    all_steps     = [r[1] for r in results]
    all_td_losses = [r[2] for r in results]

    total_time = time.time() - start_time
    print(f'\n=== All training complete ({total_time / 60:.1f} min) ===')

    # --- pkl 저장 ---
    result_path = os.path.join(results_dir, 'DQN_MultiSeed_Result.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump({'all_rewards': all_rewards, 'all_steps': all_steps,
                     'all_td_losses': all_td_losses}, f)
    print(f'Saved: {result_path}')

    # --- csv 저장 ---
    csv_path = os.path.join(results_dir, 'DQN_MultiSeed_Result.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'episode', 'total_steps', 'reward', 'td_loss'])
        for seed_idx, (rewards_s, steps_s, losses_s) in enumerate(
                zip(all_rewards, all_steps, all_td_losses), start=1):
            for ep, (s, r, l) in enumerate(zip(steps_s, rewards_s, losses_s), start=1):
                writer.writerow([seed_idx, ep, s, r, l])
    print(f'Saved: {csv_path}')

    # --- 요약 출력 ---
    print(f'Final mean reward (last 10 ep avg across seeds): '
          f'{np.nanmean([np.mean(r[-10:]) for r in all_rewards if r]):.1f}')
    print(f'Target score: {target_score}')
    print(f'\nTo generate figures:')
    print(f'  python scripts/plot_metrics.py '
          f'--csv {os.path.join(results_dir, "DQN_MultiSeed_Result.csv")} '
          f'--figures-dir figures/dqn')


if __name__ == '__main__':
    main()
