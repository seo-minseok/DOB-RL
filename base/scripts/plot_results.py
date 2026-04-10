"""
plot_results.py — 학습 결과 시각화
규칙: 하나의 이미지 파일 = 하나의 figure (subplot 금지)
Usage:
  python scripts/plot_results.py --log-dir ./logs --seed 1
  python scripts/plot_results.py --results-dir ./results --multi-seed
"""
import argparse
import os
import pickle

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
    parser.add_argument('--log-dir', type=str, default=None,
                        help='단일 시드 로그 디렉토리 (logs/)')
    parser.add_argument('--seed', type=int, default=1,
                        help='단일 시드 번호')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='figure 저장 디렉토리 (기본: 자동 설정)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.log_dir:
        log_path    = os.path.join(args.log_dir, f'seed_{args.seed}_result.pkl')
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(args.log_dir), 'figures')
        plot_single_seed(log_path, figures_dir)
    else:
        print('Usage:')
        print('  Single seed: python scripts/plot_results.py --log-dir ./logs --seed 1')


if __name__ == '__main__':
    main()
