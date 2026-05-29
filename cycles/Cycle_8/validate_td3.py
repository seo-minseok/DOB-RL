"""
validate_td3.py — real_ratio=1.0 (vanilla DDPG) 검증용 스크립트
시드별 독립 subprocess 실행. 각 subprocess 내부에서 threading.Queue로 CSV 순서 보장.

Usage:
  python validate_td3.py            # seed 1, 2 병렬 subprocess
  python validate_td3.py --seed 1   # 단일 시드 (subprocess 내부 호출)
"""
import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

SEEDS        = [1, 2]
NUM_EPISODES = 500
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR   = os.path.join(SCRIPT_DIR, 'results', 'validate_td3_run')

METRIC_KEYS = [
    'nominal_error_avg', 'residual_error_avg', 'dhat_norm_avg', 'uncertainty_avg',
    'res_net_loss', 'rbf_loss', 'td_loss_avg', 'episode_length',
    'q_pred_avg', 'target_q_avg', 'raw_episode_reward',
]


def run_single_seed(seed: int):
    """
    단일 시드 학습.
    train_DOB_core를 별도 thread에서 실행하고,
    메인 thread가 queue에서 꺼내 CSV에 순서대로 기록.
    """
    sys.path.insert(0, SCRIPT_DIR)
    from dob_mbrl.training import train_DOB_core, DOBMBRLConfig

    cfg = DOBMBRLConfig()
    cfg.real_ratio = 1.0

    ckpt_dir = os.path.join(RESULT_DIR, f'seed_{seed}')
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_path = os.path.join(RESULT_DIR, f'seed_{seed}_progress.csv')

    result_q = queue.Queue()   # threading.Queue — 순서 보장, Windows 안정

    def train_fn():
        train_DOB_core(
            run_idx        = seed,
            num_episodes   = NUM_EPISODES,
            result_queue   = result_q,
            checkpoint_dir = ckpt_dir,
            cfg            = cfg,
        )
        result_q.put(None)   # 종료 신호

    train_thread = threading.Thread(target=train_fn, daemon=True)
    train_thread.start()

    # 메인 thread: queue에서 꺼내 CSV 순서대로 기록
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'episode', 'total_steps', 'reward', 'ep_steps'] + METRIC_KEYS)
        f.flush()

        ep_count = 0
        while True:
            item = result_q.get()
            if item is None:
                break
            ep_count += 1
            writer.writerow(
                [item['run_idx'], item['ep_idx'], item['step'], item['reward'],
                 item.get('ep_steps', float('nan'))]
                + [item.get(k, float('nan')) for k in METRIC_KEYS]
            )
            f.flush()
            if ep_count % 10 == 0:
                print(f'[Seed {seed}] ep {ep_count}/{NUM_EPISODES}  '
                      f'reward={item["reward"]:.1f}', flush=True)

    train_thread.join()

    rewards = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row['reward']))

    last10 = float(np.mean(rewards[-10:])) if len(rewards) >= 10 else float('nan')
    print(f'[Seed {seed}] Done. last-10 avg={last10:.1f}  → {csv_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.seed is not None:
        # subprocess 내부: 단일 시드 실행
        run_single_seed(args.seed)
    else:
        # 메인: 각 시드를 독립 subprocess로 실행
        print('=' * 60)
        print(f'Cycle 7 DDPG validation  seeds={SEEDS}, episodes={NUM_EPISODES}')
        print(f'real_ratio=1.0  warm_start=5000  expl_noise=0.3')
        print(f'Results: {RESULT_DIR}')
        print('=' * 60, flush=True)

        procs = []
        for seed in SEEDS:
            cmd = [sys.executable, __file__, '--seed', str(seed)]
            p = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
            procs.append((seed, p))
            print(f'[Seed {seed}] subprocess started (pid={p.pid})', flush=True)

        for seed, p in procs:
            p.wait()
            print(f'[Seed {seed}] subprocess finished (returncode={p.returncode})', flush=True)

        print('\n=== All seeds done ===')
        for seed in SEEDS:
            csv_path = os.path.join(RESULT_DIR, f'seed_{seed}_progress.csv')
            if os.path.exists(csv_path):
                data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
                if data.ndim == 2 and len(data) >= 10:
                    last10 = np.mean(data[-10:, 3])
                    print(f'  Seed {seed}: {len(data)} episodes, last-10 avg={last10:.1f}')
                else:
                    print(f'  Seed {seed}: CSV rows insufficient')
            else:
                print(f'  Seed {seed}: CSV not found')
