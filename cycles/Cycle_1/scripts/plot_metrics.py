"""
plot_metrics.py — 新 지표(CSV) 기반 multi-seed figure 생성
규칙: 하나의 이미지 파일 = 하나의 figure (subplot 금지)

Usage:
  python scripts/plot_metrics.py --csv ./results/DOB_MBRL_MultiSeed_Result.csv --figures-dir ./figures/baseline
  python scripts/plot_metrics.py --csv ./results/DOB_MBRL_MultiSeed_Result_Ablation.csv --figures-dir ./figures/ablation
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict


def save_figure(fig, path: str):
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
    """seed별 값을 episode 축으로 mean±std 계산. NaN은 제외."""
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


METRIC_META = {
    'reward': {
        'label': 'Cumulative Reward',
        'title': 'Episode Reward — Mean ± Std',
        'fname': 'reward_multiseed.png',
        'hline': 480.0,
    },
    'nominal_error_avg': {
        'label': '‖dx_real − dx_nom‖₂',
        'title': 'Nominal Model Error (avg per episode)',
        'fname': 'nominal_error_avg.png',
    },
    'residual_error_avg': {
        'label': '‖(dx_real − dx_nom) − F·dx_res‖₂',
        'title': 'Residual Error after ResNet (avg per episode)',
        'fname': 'residual_error_avg.png',
    },
    'dhat_norm_avg': {
        'label': '‖dhat‖₂',
        'title': 'DOB Estimate Norm (avg per episode)',
        'fname': 'dhat_norm_avg.png',
    },
    'uncertainty_avg': {
        'label': '‖Fpinv·e − dx_res‖₂',
        'title': 'Uncertainty (avg per episode)',
        'fname': 'uncertainty_avg.png',
        'hline': 0.1,  # uncertainty_threshold
    },
    'res_net_loss': {
        'label': 'MSE Loss',
        'title': 'ResidualDxNet Training Loss (per episode)',
        'fname': 'res_net_loss.png',
    },
    'rbf_loss': {
        'label': 'MSE Loss',
        'title': 'Uncertainty RBF Training Loss (per episode)',
        'fname': 'rbf_loss.png',
    },
    'td_loss_avg': {
        'label': 'TD MSE Loss',
        'title': 'TD Loss — Mean ± Std (avg per episode)',
        'fname': 'td_loss_avg.png',
    },
    'episode_length': {
        'label': 'Steps',
        'title': 'Episode Length — Mean ± Std',
        'fname': 'episode_length.png',
        'hline': 500.0,
    },
    'epsilon': {
        'label': 'ε',
        'title': 'Epsilon (end of episode)',
        'fname': 'epsilon.png',
    },
}


def plot_metric(data: dict, col: str, figures_dir: str, num_seeds: int):
    meta = METRIC_META.get(col, {
        'label': col,
        'title': col,
        'fname': f'{col}.png',
    })

    episodes, mean, std = compute_mean_std(data, col)

    # NaN이 전부인 구간 제거 (warm-up NaN 처리)
    valid = ~np.isnan(mean)
    if not valid.any():
        print(f'Skip {col}: all NaN')
        return

    ep_v   = episodes[valid]
    mean_v = mean[valid]
    std_v  = std[valid]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
    ax.fill_between(ep_v, mean_v - std_v, mean_v + std_v,
                    color='steelblue', alpha=0.2, linewidth=0, label='±1 Std')
    ax.plot(ep_v, mean_v, color='steelblue', linewidth=2.0, label='Mean')

    if 'hline' in meta:
        ax.axhline(y=meta['hline'], color='red', linestyle='--',
                   linewidth=1.5, label=f'{meta["hline"]}')

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel(meta['label'], fontsize=12, fontweight='bold')
    ax.set_title(f'{meta["title"]} ({num_seeds} seeds)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True)

    save_figure(fig, os.path.join(figures_dir, meta['fname']))


def plot_compare_reward(baseline_csv: str, ablation_csv: str, figures_dir: str):
    """Baseline(파랑) vs Ablation(빨강) reward 오버레이 figure."""
    b_data = load_csv(baseline_csv)
    a_data = load_csv(ablation_csv)

    b_ep, b_mean, b_std = compute_mean_std(b_data, 'reward')
    a_ep, a_mean, a_std = compute_mean_std(a_data, 'reward')

    b_valid = ~np.isnan(b_mean)
    a_valid = ~np.isnan(a_mean)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')

    ax.fill_between(b_ep[b_valid], (b_mean - b_std)[b_valid], (b_mean + b_std)[b_valid],
                    color='steelblue', alpha=0.2, linewidth=0)
    ax.plot(b_ep[b_valid], b_mean[b_valid],
            color='steelblue', linewidth=2.0, label=f'Baseline (n={len(b_data)})')

    ax.fill_between(a_ep[a_valid], (a_mean - a_std)[a_valid], (a_mean + a_std)[a_valid],
                    color='crimson', alpha=0.2, linewidth=0)
    ax.plot(a_ep[a_valid], a_mean[a_valid],
            color='crimson', linewidth=2.0, label=f'Ablation (n={len(a_data)})')

    ax.axhline(y=480.0, color='gray', linestyle='--', linewidth=1.5, label='480')

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title('Episode Reward — Baseline vs Ablation (Mean ± Std)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True)

    os.makedirs(figures_dir, exist_ok=True)
    save_figure(fig, os.path.join(figures_dir, 'compare_baseline_vs_ablation_reward.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None,
                        help='결과 CSV 경로 (단일 조건 figure 생성 시)')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='figure 저장 디렉토리')
    parser.add_argument('--compare', action='store_true',
                        help='Baseline vs Ablation 비교 오버레이 figure 생성')
    parser.add_argument('--baseline-csv', type=str, default=None,
                        help='--compare 시 baseline CSV 경로')
    parser.add_argument('--ablation-csv', type=str, default=None,
                        help='--compare 시 ablation CSV 경로')
    args = parser.parse_args()

    if args.compare:
        if not args.baseline_csv or not args.ablation_csv:
            parser.error('--compare requires --baseline-csv and --ablation-csv')
        figures_dir = args.figures_dir or os.path.join(
            os.path.dirname(os.path.dirname(args.baseline_csv)), 'figures')
        plot_compare_reward(args.baseline_csv, args.ablation_csv, figures_dir)
        return

    if not args.csv:
        parser.error('--csv is required when not using --compare')

    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.dirname(args.csv)), 'figures', 'baseline')
    os.makedirs(figures_dir, exist_ok=True)

    data = load_csv(args.csv)
    num_seeds = len(data)
    print(f'Loaded {num_seeds} seeds from {args.csv}')

    for col in METRIC_META:
        plot_metric(data, col, figures_dir, num_seeds)


if __name__ == '__main__':
    main()
