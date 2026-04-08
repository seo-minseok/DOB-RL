"""
plot_metrics.py — 로그 → metric별 개별 figure 생성
규칙: 하나의 이미지 파일 = 하나의 figure (subplot 금지)
Usage:
  python plot_metrics.py --log-path Cycle_1/logs/seed_1_result.pkl --figures-dir Cycle_1/figures
  python plot_metrics.py --results-path Cycle_1/results/DOB_MBRL_MultiSeed_Result.pkl --figures-dir Cycle_1/figures
"""
import argparse
import os
import pickle
import sys

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
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_single(log_path: str, figures_dir: str):
    with open(log_path, 'rb') as f:
        data = pickle.load(f)

    rewards = data['rewards']
    steps   = data['steps']

    # reward_curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
    ax.plot(range(1, len(rewards) + 1), rewards, 'k-', linewidth=1.0)
    ax.axhline(y=480, color='red', linestyle='--', linewidth=1.5, label='Target (480)')
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title('Episode Reward Curve', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True)
    save_figure(fig, os.path.join(figures_dir, 'reward_curve.png'))

    # total_steps
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
    ax.plot(range(1, len(steps) + 1), steps, 'b-', linewidth=1.0)
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Steps (cumulative)', fontsize=12, fontweight='bold')
    ax.set_title('Total Environment Steps', fontsize=13, fontweight='bold')
    ax.grid(True)
    save_figure(fig, os.path.join(figures_dir, 'total_steps.png'))

    # smoothed
    if len(rewards) >= 10:
        smoothed = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
        ax.plot(range(10, len(rewards) + 1), smoothed, 'k-', linewidth=1.5)
        ax.axhline(y=480, color='red', linestyle='--', linewidth=1.5, label='Target (480)')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reward (10-ep avg)', fontsize=12, fontweight='bold')
        ax.set_title('Smoothed Reward (10-ep moving average)', fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(True)
        save_figure(fig, os.path.join(figures_dir, 'reward_smoothed.png'))


def plot_multi(results_path: str, figures_dir: str, target_score: float = 480.0):
    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    all_rewards = data['all_rewards']
    all_steps   = data['all_steps']
    num_runs    = len(all_rewards)

    max_total_step = max(max(s) for s in all_steps if s)
    common_steps   = np.linspace(0, max_total_step, 1000)
    interp_rewards = np.full((num_runs, 1000), np.nan, dtype=np.float64)

    for i in range(num_runs):
        x_run = np.array([0] + list(all_steps[i]),   dtype=np.float64)
        y_run = np.array([0] + list(all_rewards[i]), dtype=np.float64)
        _, uid = np.unique(x_run, return_index=True)
        x_run, y_run = x_run[uid], y_run[uid]
        if len(x_run) < 2:
            continue
        vals = np.interp(common_steps, x_run, y_run, left=np.nan, right=np.nan)
        vals[common_steps > x_run[-1]] = y_run[-1]
        interp_rewards[i, :] = vals

    mean_curve  = np.nanmean(interp_rewards, axis=0)
    std_curve   = np.nanstd( interp_rewards, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
    ax.fill_between(common_steps, mean_curve - std_curve, mean_curve + std_curve,
                    color='black', alpha=0.2, linewidth=0, label='±1 Std')
    ax.plot(common_steps, mean_curve, 'k-', linewidth=2.5, label='Mean Reward')
    ax.axhline(y=target_score, color='black', linestyle='--',
               linewidth=1.5, label=f'Target ({target_score})')
    ax.set_xlabel('Total Environmental Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title(f'DOB-MBRL Multi-Seed ({num_runs} runs) — Mean ± Std',
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True)
    save_figure(fig, os.path.join(figures_dir, 'multiseed_mean_std.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-path',     type=str, default=None)
    parser.add_argument('--results-path', type=str, default=None)
    parser.add_argument('--figures-dir',  type=str, required=True)
    args = parser.parse_args()

    if args.log_path:
        plot_single(args.log_path, args.figures_dir)
    elif args.results_path:
        plot_multi(args.results_path, args.figures_dir)
    else:
        print('Provide --log-path or --results-path')
        sys.exit(1)


if __name__ == '__main__':
    main()
