"""
run_multi_seed.py — 멀티 시드 병렬 학습 진입점
Usage:
  python run_multi_seed.py --num-seeds 8 --num-gpus 4
  python run_multi_seed.py --num-seeds 8 --num-gpus 4 --checkpoint-dir ./checkpoints
"""
import argparse
import csv
import multiprocessing
import os
import pickle
import time

import numpy as np
import torch

from mbpo.training import train_MBPO_core, MBPOConfig

METRIC_KEYS = [
    'td_loss_avg', 'q_pred_avg', 'target_q_avg', 'alpha',
    'model_val_loss', 'model_error_avg', 'rollout_mean_length', 'model_pool_size',
    'episode_length', 'env_return', 'eval_reward_avg',
]


def _run_single(args):
    run_idx, checkpoint_dir, results_dir, gpu_id = args
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'[Run {run_idx:02d}] device={device}  MBPO training started...')
    cfg = MBPOConfig()
    rewards, steps, metrics = train_MBPO_core(
        run_idx        = run_idx,
        checkpoint_dir = checkpoint_dir,
        results_dir    = results_dir,
        cfg            = cfg,
        device         = device,
    )
    print(f'[Run {run_idx:02d}] Done.')
    return rewards, steps, metrics


def parse_args():
    parser = argparse.ArgumentParser(description='MBPO multi-seed training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/default',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--num-seeds', type=int, default=8, help='병렬 실행 시드 수')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='사용할 GPU 수 (seeds를 round-robin으로 배분)')
    parser.add_argument('--result-name', type=str, default=None,
                        help='출력 파일 기본 이름 (확장자 제외)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = MBPOConfig()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results', 'default')
    os.makedirs(results_dir, exist_ok=True)

    num_runs = args.num_seeds
    num_gpus = args.num_gpus
    print('=' * 60)
    print(f'MBPO Multi-Seed Training  (runs={num_runs}, gpus={num_gpus}, '
          f'total_steps={cfg.total_steps})')
    print(f'real_ratio: {cfg.real_ratio}  |  num_networks: {cfg.num_networks}  '
          f'|  num_elites: {cfg.num_elites}')
    print(f'results_dir: {results_dir}')
    print('=' * 60)
    start_time = time.time()

    # round-robin GPU 할당: seed i → GPU (i-1) % num_gpus
    args_list = [
        (i + 1, args.checkpoint_dir, results_dir, i % num_gpus)
        for i in range(num_runs)
    ]

    num_workers = min(num_runs, multiprocessing.cpu_count())
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_run_single, args_list)
    except (OSError, PermissionError) as exc:
        print(f'Parallel unavailable ({exc}). Falling back to sequential.')
        results = [_run_single(a) for a in args_list]

    all_rewards = [r[0] for r in results]
    all_steps   = [r[1] for r in results]
    all_metrics = [r[2] for r in results]

    total_time = time.time() - start_time
    print(f'\n=== All training complete ({total_time / 60:.1f} min) ===')

    base_name   = args.result_name or 'MBPO_MultiSeed_Result'
    result_path = os.path.join(results_dir, base_name + '.pkl')
    csv_path    = os.path.join(results_dir, base_name + '.csv')

    with open(result_path, 'wb') as f:
        pickle.dump({'all_rewards': all_rewards, 'all_steps': all_steps,
                     'all_metrics': all_metrics}, f)
    print(f'Saved: {result_path}')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'episode', 'total_steps', 'reward'] + METRIC_KEYS)
        for seed_idx, (rewards_s, steps_s, metrics_s) in enumerate(
                zip(all_rewards, all_steps, all_metrics), start=1):
            for ep, (s, r, m) in enumerate(zip(steps_s, rewards_s, metrics_s), start=1):
                writer.writerow(
                    [seed_idx, ep, s, r] + [m.get(k, float('nan')) for k in METRIC_KEYS]
                )
    print(f'Saved: {csv_path}')

    final_mean = np.nanmean([np.mean(r[-10:]) for r in all_rewards if r])
    print(f'Final mean reward (last 10 ep avg across seeds): {final_mean:.1f}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
