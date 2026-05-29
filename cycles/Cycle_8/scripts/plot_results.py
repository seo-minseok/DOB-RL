"""
plot_results.py — 학습 결과 시각화
규칙: 하나의 이미지 파일 = 하나의 figure (subplot 금지)
Usage:
  python scripts/plot_results.py --log-dir ./logs --seed 1
  python scripts/plot_results.py --csv ./results/DOB_MBRL_MultiSeed_Result.csv --figures-dir ./figures/baseline
"""
import argparse
import csv
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METRIC_META = {
    'reward':             ('Cumulative Reward',     'Episode Reward'),
    'episode_length':     ('Steps',                 'Episode Length'),
    'nominal_error_avg':  ('Error',                 'Nominal Error (avg)'),
    'residual_error_avg': ('Error',                 'Residual Error (avg)'),
    'dhat_norm_avg':      ('||d_hat||',             'DOB Estimate Norm (avg)'),
    'uncertainty_avg':    ('Uncertainty',           'Uncertainty (avg)'),
    'res_net_loss':       ('Loss',                  'ResNet Loss'),
    'rbf_loss':           ('Loss',                  'RBF Loss'),
    'td_loss_avg':        ('TD Loss',               'TD Loss (avg)'),
    'epsilon':            ('Epsilon',               'Exploration Epsilon'),
    'buffer_uncert_avg':  ('Uncertainty',           'Buffer Uncertainty (avg)'),
    'sampled_uncert_avg': ('Uncertainty',           'Sampled Uncertainty (avg)'),
    'fresh_uncert_avg':   ('Uncertainty',           'Fresh Uncertainty Target (avg)'),
    'rollout_uncert_avg': ('Uncertainty',           'Rollout Uncertainty (avg)'),
}


def load_csv(csv_path: str):
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(row['seed'])
            for k, v in row.items():
                if k == 'seed' or k is None:
                    continue
                try:
                    val = v if isinstance(v, str) else str(v)
                    data[seed][k].append(float(val) if val not in ('nan', '', 'None') else float('nan'))
                except (ValueError, TypeError):
                    data[seed][k].append(float('nan'))
    return data


def compute_mean_std(data: dict, col: str):
    seeds = sorted(data.keys())
    max_ep = max(len(data[s][col]) for s in seeds)
    matrix = np.full((len(seeds), max_ep), np.nan)
    for i, s in enumerate(seeds):
        vals = data[s][col]
        matrix[i, :len(vals)] = vals
    mean = np.nanmean(matrix, axis=0)
    std  = np.nanstd(matrix, axis=0)
    return np.arange(1, max_ep + 1), mean, std


def plot_multiseed_csv(csv_path: str, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    data = load_csv(csv_path)
    num_seeds = len(data)
    print(f'Loaded {num_seeds} seeds from {csv_path}')

    for col, (ylabel, title) in METRIC_META.items():
        episodes, mean, std = compute_mean_std(data, col)
        valid = ~np.isnan(mean)
        if not valid.any():
            print(f'Skip {col}: all NaN')
            continue

        ep_v, mean_v, std_v = episodes[valid], mean[valid], std[valid]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
        ax.fill_between(ep_v, mean_v - std_v, mean_v + std_v,
                        color='steelblue', alpha=0.2, linewidth=0, label='±1 Std')
        ax.plot(ep_v, mean_v, color='steelblue', linewidth=2.0, label='Mean')
        if col == 'reward':
            ax.axhline(y=480, color='red', linestyle='--', linewidth=1.5, label='Target (480)')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} — Mean ± Std ({num_seeds} seeds)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)
        save_figure(fig, os.path.join(figures_dir, f'{col}.png'))



def plot_per_seed_csv(csv_path: str, figures_dir: str):
    """시드별 개별 figure 생성 → figures_dir/seed_{N}/ 하위에 저장."""
    data = load_csv(csv_path)
    seeds = sorted(data.keys())
    print(f'Generating per-seed figures for {len(seeds)} seeds...')

    for seed in seeds:
        seed_dir = os.path.join(figures_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        seed_data = data[seed]

        for col, (ylabel, title) in METRIC_META.items():
            vals = seed_data.get(col, [])
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            valid = ~np.isnan(arr)
            if not valid.any():
                continue

            episodes = np.arange(1, len(arr) + 1)
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
            ax.plot(episodes[valid], arr[valid], color='steelblue', linewidth=1.5)
            if col == 'reward':
                ax.axhline(y=480, color='red', linestyle='--', linewidth=1.5, label='Target (480)')
                ax.legend()
            ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(f'{title} — Seed {seed}', fontsize=13, fontweight='bold')
            ax.grid(True)
            save_figure(fig, os.path.join(seed_dir, f'{col}.png'))


def save_figure(fig, path: str):
    """하나의 figure를 하나의 파일로 저장. subplot 여부 검증."""
    ax_list = fig.get_axes()
    assert len(ax_list) == 1, (
        f"Figure must have exactly 1 Axes (got {len(ax_list)}). "
        "Each metric must be saved to a separate file."
    )
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_single_seed(log_path: str, figures_dir: str):
    with open(log_path, 'rb') as f:
        data = pickle.load(f)

    rewards = data['rewards']
    steps   = data['steps']

    os.makedirs(figures_dir, exist_ok=True)

    # Figure 1: Episode Reward
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
    ax.plot(range(1, len(rewards) + 1), rewards, 'k-', linewidth=1.0)
    ax.axhline(y=480, color='red', linestyle='--', linewidth=1.5, label='Target (480)')
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title('Episode Reward Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True)
    save_figure(fig, os.path.join(figures_dir, 'reward_curve.png'))

    # Figure 2: Total Steps
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
    ax.plot(range(1, len(steps) + 1), steps, 'b-', linewidth=1.0)
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Steps (cumulative)', fontsize=12, fontweight='bold')
    ax.set_title('Total Environment Steps', fontsize=13, fontweight='bold')
    ax.grid(True)
    save_figure(fig, os.path.join(figures_dir, 'total_steps.png'))

    # Figure 3: Smoothed Reward (10-ep moving avg)
    if len(rewards) >= 10:
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
        ax.plot(range(10, len(rewards) + 1), smoothed, 'k-', linewidth=1.5)
        ax.axhline(y=480, color='red', linestyle='--', linewidth=1.5, label='Target (480)')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward (10-ep avg)', fontsize=12, fontweight='bold')
        ax.set_title('Smoothed Reward (10-ep moving average)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)
        save_figure(fig, os.path.join(figures_dir, 'reward_smoothed.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='DOB-MBRL plot results')
    parser.add_argument('--csv', type=str, default=None,
                        help='멀티시드 결과 CSV 경로')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='단일 시드 로그 디렉토리 (logs/)')
    parser.add_argument('--seed', type=int, default=1,
                        help='단일 시드 번호')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='figure 저장 디렉토리 (기본: 자동 설정)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.csv:
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(os.path.dirname(args.csv)), 'figures', 'baseline')
        plot_multiseed_csv(args.csv, figures_dir)
        plot_per_seed_csv(args.csv, figures_dir)
    elif args.log_dir:
        log_path    = os.path.join(args.log_dir, f'seed_{args.seed}_result.pkl')
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(args.log_dir), 'figures')
        plot_single_seed(log_path, figures_dir)
    else:
        print('Usage:')
        print('  Multi-seed: python scripts/plot_results.py --csv ./results/DOB_MBRL_MultiSeed_Result.csv')
        print('  Single seed: python scripts/plot_results.py --log-dir ./logs --seed 1')


if __name__ == '__main__':
    main()
