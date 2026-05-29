"""
run_multi_seed_test.py — 멀티 시드 평가 진입점
저장된 체크포인트를 순차 로드해 greedy 정책으로 평가하고
스텝 단위 CSV를 시드별로 저장한다.

Usage:
  python run_multi_seed_test.py                         # baseline, seed 1-16, 30 ep
  python run_multi_seed_test.py --variant ablation
  python run_multi_seed_test.py --seeds 1 2 3 --num-test-episodes 50
"""
import argparse
import csv
import os
import time

import numpy as np

from test_DOB_core import test_DOB_core
from dob_mbrl.training.config import DOBMBRLConfig


# 스텝 로그 컬럼 순서 (CSV 헤더)
STEP_LOG_COLUMNS = [
    'seed', 'episode', 'step',
    'obs_x', 'obs_xdot', 'obs_theta', 'obs_thetadot',
    'action_idx', 'action_force',
    'reward', 'done',
    'dx_real_xdot', 'dx_real_thetadot',
    'dx_nom_xdot', 'dx_nom_thetadot',
    'dx_res_0', 'dx_res_1',
    'nominal_error', 'residual_error',
    'dhat_0', 'dhat_1', 'dhat_norm',
    'uncertainty_0', 'uncertainty_1', 'uncertainty_mag',
]


def parse_args():
    parser = argparse.ArgumentParser(description='DOB-MBRL multi-seed test')
    parser.add_argument(
        '--variant', type=str, default='baseline',
        choices=['baseline', 'ablation'],
        help='평가할 체크포인트 종류 (default: baseline)',
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='체크포인트 디렉토리 (미지정 시 ./checkpoints/{variant} 사용)',
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=list(range(1, 17)),
        help='평가할 시드 번호 목록 (default: 1-16)',
    )
    parser.add_argument(
        '--num-test-episodes', type=int, default=30,
        help='시드당 평가 에피소드 수 (default: 30)',
    )
    return parser.parse_args()


def save_step_log_csv(step_logs: list[dict], path: str):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=STEP_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(step_logs)


def print_seed_summary(run_idx: int, step_logs: list[dict], max_steps: int):
    """스텝 로그에서 에피소드 요약을 계산해 출력한다."""
    if not step_logs:
        print(f'  [Seed {run_idx:02d}] 로그 없음')
        return

    episodes = {}
    for row in step_logs:
        ep = row['episode']
        if ep not in episodes:
            episodes[ep] = {'reward': 0.0, 'length': 0,
                            'nominal_errors': [], 'residual_errors': []}
        episodes[ep]['reward']           += row['reward']
        episodes[ep]['length']           += 1
        episodes[ep]['nominal_errors'].append(row['nominal_error'])
        episodes[ep]['residual_errors'].append(row['residual_error'])

    rewards   = [v['reward']  for v in episodes.values()]
    lengths   = [v['length']  for v in episodes.values()]
    successes = [l == max_steps for l in lengths]

    print(f'  [Seed {run_idx:02d}] '
          f'reward={np.mean(rewards):.2f}±{np.std(rewards):.2f}  '
          f'length={np.mean(lengths):.1f}  '
          f'success={sum(successes)}/{len(successes)}')


def main():
    args = parse_args()

    script_dir    = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        script_dir, 'checkpoints', args.variant
    )
    results_dir = os.path.join(script_dir, 'results', 'test', args.variant)
    os.makedirs(results_dir, exist_ok=True)

    cfg = DOBMBRLConfig()

    print('=' * 60)
    print(f'DOB-MBRL Multi-Seed Test')
    print(f'  variant          : {args.variant}')
    print(f'  checkpoint_dir   : {checkpoint_dir}')
    print(f'  seeds            : {args.seeds}')
    print(f'  test_episodes    : {args.num_test_episodes}')
    print(f'  results_dir      : {results_dir}')
    print('=' * 60)

    start_time = time.time()
    all_step_logs = []   # 전 시드 합산 (summary CSV용)

    for run_idx in args.seeds:
        t0 = time.time()
        print(f'[Seed {run_idx:02d}] 평가 시작...')

        try:
            step_logs = test_DOB_core(
                run_idx           = run_idx,
                checkpoint_dir    = checkpoint_dir,
                num_test_episodes = args.num_test_episodes,
                cfg               = cfg,
            )
        except FileNotFoundError as e:
            print(f'  경고: {e} — 건너뜀')
            continue

        # 시드별 CSV 저장
        seed_csv = os.path.join(results_dir, f'Seed{run_idx}_step_log.csv')
        save_step_log_csv(step_logs, seed_csv)

        elapsed = time.time() - t0
        print(f'  저장: {seed_csv}  ({elapsed:.1f}s)')
        print_seed_summary(run_idx, step_logs, cfg.max_steps_per_ep)

        all_step_logs.extend(step_logs)

    # 전체 시드 합산 CSV 저장
    summary_csv = os.path.join(results_dir, 'all_seeds_step_log.csv')
    save_step_log_csv(all_step_logs, summary_csv)
    print(f'\n전체 합산 저장: {summary_csv}')

    # 전체 요약 출력
    if all_step_logs:
        _print_overall_summary(all_step_logs, cfg.max_steps_per_ep, args.variant)

    total_time = time.time() - start_time
    print(f'\n=== 평가 완료 ({total_time:.1f}s) ===')


def _print_overall_summary(all_step_logs: list[dict], max_steps: int, variant: str):
    """전체 시드 × 에피소드 집계 요약."""
    ep_rewards  = {}
    ep_lengths  = {}

    for row in all_step_logs:
        key = (row['seed'], row['episode'])
        ep_rewards.setdefault(key, 0.0)
        ep_rewards[key] += row['reward']
        ep_lengths[key]  = max(ep_lengths.get(key, 0), row['step'])

    rewards   = list(ep_rewards.values())
    lengths   = list(ep_lengths.values())
    successes = [l == max_steps for l in lengths]

    print(f'\n[{variant.upper()} 전체 요약]')
    print(f'  에피소드 수     : {len(rewards)}')
    print(f'  reward mean±std : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}')
    print(f'  length mean     : {np.mean(lengths):.1f}')
    print(f'  success rate    : {sum(successes)}/{len(successes)} '
          f'({100*sum(successes)/len(successes):.1f}%)')


if __name__ == '__main__':
    main()
