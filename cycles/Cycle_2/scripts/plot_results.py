"""
plot_results.py — 학습 결과 시각화
규칙: 하나의 이미지 파일 = 하나의 figure (subplot 금지)
Usage:
  python scripts/plot_results.py --csv ./results/DOB_MBRL_MultiSeed_Result.csv --figures-dir ./figures/baseline
  python scripts/plot_results.py --log-dir ./logs --seed 1
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


def load_csv(csv_path: str):
    """CSV → {seed: {col: list}} 구조로 로드."""
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(row['seed'])
            for k, v in row.items():
                if k == 'seed':
                    continue
                try:
                    data[seed][k].append(float(v) if v != 'nan' else float('nan'))
                except ValueError:
                    data[seed][k].append(float('nan'))
    return data


def compute_mean_std(data: dict, col: str):
    """seed별 값을 episode 축으로 mean±std 계산."""
    seeds = sorted(data.keys())
    max_ep = max(len(data[s][col]) for s in seeds)
    matrix = np.full((len(seeds), max_ep), np.nan)
    for i, s in enumerate(seeds):
        vals = data[s][col]
        matrix[i, :len(vals)] = vals
    mean = np.nanmean(matrix, axis=0)
    std  = np.nanstd(matrix, axis=0)
    episodes = np.arange(1, max_ep + 1)
    return episodes, mean, std


def plot_multiseed_csv(csv_path: str, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    data = load_csv(csv_path)
    num_seeds = len(data)
    print(f'Loaded {num_seeds} seeds from {csv_path}')

    # episode_length 계산: total_steps 차분
    for s in data:
        ts = data[s]['total_steps']
        lengths = [ts[0]] + [ts[i] - ts[i-1] for i in range(1, len(ts))]
        data[s]['episode_length'] = lengths

    metrics = [
        ('reward', 'Cumulative Reward', 'Episode Reward — Mean ± Std', 300.0, 'reward_multiseed.png'),
        ('episode_length', 'Steps', 'Episode Length — Mean ± Std', 500.0, 'episode_length.png'),
    ]

    for col, ylabel, title, hline, fname in metrics:
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
        ax.axhline(y=hline, color='red', linestyle='--', linewidth=1.5, label=str(hline))
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} ({num_seeds} seeds)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)
        save_figure(fig, os.path.join(figures_dir, fname))


def plot_single_seed(log_path: str, figures_dir: str):
    with open(log_path, 'rb') as f:
        data = pickle.load(f)

    rewards = data['rewards']
    steps   = data['steps']

    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
    ax.plot(range(1, len(rewards) + 1), rewards, 'k-', linewidth=1.0)
    ax.axhline(y=300, color='red', linestyle='--', linewidth=1.5, label='Target (300)')
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title('Episode Reward Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True)
    save_figure(fig, os.path.join(figures_dir, 'reward_curve.png'))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
    ax.plot(range(1, len(steps) + 1), steps, 'b-', linewidth=1.0)
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Steps (cumulative)', fontsize=12, fontweight='bold')
    ax.set_title('Total Environment Steps', fontsize=13, fontweight='bold')
    ax.grid(True)
    save_figure(fig, os.path.join(figures_dir, 'total_steps.png'))

    if len(rewards) >= 10:
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
        ax.plot(range(10, len(rewards) + 1), smoothed, 'k-', linewidth=1.5)
        ax.axhline(y=300, color='red', linestyle='--', linewidth=1.5, label='Target (300)')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward (10-ep avg)', fontsize=12, fontweight='bold')
        ax.set_title('Smoothed Reward (10-ep moving average)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)
        save_figure(fig, os.path.join(figures_dir, 'reward_smoothed.png'))


def main():
    parser = argparse.ArgumentParser(description='DOB-MBRL plot results')
    parser.add_argument('--csv', type=str, default=None,
                        help='멀티시드 결과 CSV 경로')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='단일 시드 로그 디렉토리 (logs/)')
    parser.add_argument('--seed', type=int, default=1,
                        help='단일 시드 번호')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='figure 저장 디렉토리')
    args = parser.parse_args()

    if args.csv:
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(os.path.dirname(args.csv)), 'figures', 'baseline')
        plot_multiseed_csv(args.csv, figures_dir)
    elif args.log_dir:
        log_path    = os.path.join(args.log_dir, f'seed_{args.seed}_result.pkl')
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(args.log_dir), 'figures')
        plot_single_seed(log_path, figures_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
