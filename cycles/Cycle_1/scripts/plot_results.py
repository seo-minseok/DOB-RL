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


def plot_multi_seed(results_path: str, figures_dir: str, target_score: float = 480.0):
    with open(results_path, 'rb') as f:
        data = pickle.load(f)

    all_rewards = data['all_rewards']
    all_steps   = data['all_steps']
    num_runs    = len(all_rewards)

    os.makedirs(figures_dir, exist_ok=True)

    # Interpolation
    max_total_step = max(max(s) for s in all_steps if s)
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
        vals[common_steps > x_run[-1]] = y_run[-1]
        interp_rewards[i, :] = vals

    mean_curve  = np.nanmean(interp_rewards, axis=0)
    std_curve   = np.nanstd( interp_rewards, axis=0)
    upper_curve = mean_curve + std_curve
    lower_curve = mean_curve - std_curve

    # Figure: Mean ± Std (오버레이 — subplot 아님)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')
    ax.fill_between(common_steps, lower_curve, upper_curve,
                    color='black', alpha=0.2, linewidth=0, label='±1 Std')
    ax.plot(common_steps, mean_curve, 'k-', linewidth=2.5, label='Mean Reward')
    ax.axhline(y=target_score, color='black', linestyle='--',
               linewidth=1.5, label=f'Target ({target_score})')
    ax.set_xlabel('Total Environmental Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title(f'DOB-MBRL Multi-Seed ({num_runs} runs) — Mean ± Std',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True)
    save_figure(fig, os.path.join(figures_dir, 'multiseed_mean_std.png'))


def plot_compare(results_dir: str, figures_dir: str, target_score: float = 480.0):
    """Baseline vs Ablation mean±std 오버레이 비교 figure."""
    baseline_path = os.path.join(results_dir, 'DOB_MBRL_MultiSeed_Result.pkl')
    ablation_path = os.path.join(results_dir, 'DOB_MBRL_MultiSeed_Result_Ablation.pkl')

    with open(baseline_path, 'rb') as f:
        base_data = pickle.load(f)
    with open(ablation_path, 'rb') as f:
        abl_data = pickle.load(f)

    def interpolate(data):
        all_rewards = data['all_rewards']
        all_steps   = data['all_steps']
        num_runs    = len(all_rewards)
        max_step    = max(max(s) for s in all_steps if s)
        common      = np.linspace(0, max_step, 1000)
        interp      = np.full((num_runs, 1000), np.nan)
        for i in range(num_runs):
            x = np.array([0] + list(all_steps[i]),   dtype=np.float64)
            y = np.array([0] + list(all_rewards[i]), dtype=np.float64)
            _, uid = np.unique(x, return_index=True)
            x, y = x[uid], y[uid]
            if len(x) < 2:
                continue
            vals = np.interp(common, x, y, left=np.nan, right=np.nan)
            vals[common > x[-1]] = y[-1]
            interp[i] = vals
        return common, np.nanmean(interp, axis=0), np.nanstd(interp, axis=0)

    base_x, base_mean, base_std = interpolate(base_data)
    abl_x,  abl_mean,  abl_std  = interpolate(abl_data)

    os.makedirs(figures_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')

    ax.fill_between(base_x, base_mean - base_std, base_mean + base_std,
                    color='steelblue', alpha=0.15, linewidth=0)
    ax.plot(base_x, base_mean, color='steelblue', linewidth=2.5,
            label=f'Baseline (uncertainty-weighted, n={len(base_data["all_rewards"])})')

    ax.fill_between(abl_x, abl_mean - abl_std, abl_mean + abl_std,
                    color='red', alpha=0.15, linewidth=0)
    ax.plot(abl_x, abl_mean, color='red', linewidth=2.5,
            label=f'Ablation (uniform sampling, n={len(abl_data["all_rewards"])})')

    ax.axhline(y=target_score, color='black', linestyle='--',
               linewidth=1.5, label=f'Target ({target_score})')

    ax.set_xlabel('Total Environmental Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title('Baseline vs Ablation — Mean ± Std', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True)

    save_figure(fig, os.path.join(figures_dir, 'compare_baseline_vs_ablation.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='DOB-MBRL plot results')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='단일 시드 로그 디렉토리 (logs/)')
    parser.add_argument('--seed', type=int, default=1,
                        help='단일 시드 번호')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='멀티시드 결과 디렉토리 (results/)')
    parser.add_argument('--multi-seed', action='store_true',
                        help='멀티시드 결과 플롯')
    parser.add_argument('--ablation', action='store_true',
                        help='Ablation 결과 플롯 (DOB_MBRL_MultiSeed_Result_Ablation.pkl 사용)')
    parser.add_argument('--compare', action='store_true',
                        help='Baseline vs Ablation 비교 오버레이 figure 생성')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='figure 저장 디렉토리 (기본: 자동 설정)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare and args.results_dir:
        base_dir    = os.path.dirname(args.results_dir)
        figures_dir = args.figures_dir or os.path.join(base_dir, 'figures')
        plot_compare(args.results_dir, figures_dir)
    elif args.multi_seed and args.results_dir:
        fname        = ('DOB_MBRL_MultiSeed_Result_Ablation.pkl' if args.ablation
                        else 'DOB_MBRL_MultiSeed_Result.pkl')
        results_path = os.path.join(args.results_dir, fname)
        base_dir     = os.path.dirname(args.results_dir)
        sub          = 'ablation' if args.ablation else 'baseline'
        figures_dir  = args.figures_dir or os.path.join(base_dir, 'figures', sub)
        plot_multi_seed(results_path, figures_dir)
    elif args.log_dir:
        log_path    = os.path.join(args.log_dir, f'seed_{args.seed}_result.pkl')
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(args.log_dir), 'figures')
        plot_single_seed(log_path, figures_dir)
    else:
        print('Usage:')
        print('  Single seed: python scripts/plot_results.py --log-dir ./logs --seed 1')
        print('  Multi seed:  python scripts/plot_results.py --results-dir ./results --multi-seed')


if __name__ == '__main__':
    main()
