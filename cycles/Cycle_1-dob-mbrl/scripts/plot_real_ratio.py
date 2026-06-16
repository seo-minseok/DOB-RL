"""
plot_real_ratio.py — real_ratio 실험 결과 시각화
규칙: 하나의 이미지 파일 = 하나의 figure (subplot 금지)

Usage:
  python scripts/plot_real_ratio.py --results-dir ./results/real_ratio=1.0 --figures-dir ./figures/real_ratio=1.0
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict


METRIC_META = {
    'reward': {
        'label': 'Cumulative Reward',
        'title': 'Episode Reward',
        'fname': 'reward.png',
        'hline': 480.0,
    },
    'eval_reward_avg': {
        'label': 'Eval Reward (avg)',
        'title': 'Evaluation Reward (avg)',
        'fname': 'eval_reward_avg.png',
        'hline': 480.0,
    },
    'ep_steps': {
        'label': 'Steps',
        'title': 'Episode Length',
        'fname': 'ep_steps.png',
        'hline': 500.0,
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
        'hline': 0.1,
    },
    'res_net_loss': {
        'label': 'MSE Loss',
        'title': 'ResidualDxNet Training Loss',
        'fname': 'res_net_loss.png',
    },
    'rbf_loss': {
        'label': 'MSE Loss',
        'title': 'Uncertainty RBF Training Loss',
        'fname': 'rbf_loss.png',
    },
    'td_loss_avg': {
        'label': 'TD MSE Loss',
        'title': 'TD Loss (avg per episode)',
        'fname': 'td_loss_avg.png',
    },
    'buffer_uncert_avg': {
        'label': 'Uncertainty',
        'title': 'Buffer Uncertainty (avg)',
        'fname': 'buffer_uncert_avg.png',
    },
    'sampled_uncert_avg': {
        'label': 'Uncertainty',
        'title': 'Sampled Uncertainty (avg)',
        'fname': 'sampled_uncert_avg.png',
    },
    'fresh_uncert_avg': {
        'label': 'Uncertainty',
        'title': 'Fresh Uncertainty Target (avg)',
        'fname': 'fresh_uncert_avg.png',
    },
    'rollout_uncert_avg': {
        'label': 'Uncertainty',
        'title': 'Rollout Uncertainty (avg)',
        'fname': 'rollout_uncert_avg.png',
    },
    'rollout_pass_rate': {
        'label': 'Pass Rate',
        'title': 'Rollout Pass Rate',
        'fname': 'rollout_pass_rate.png',
    },
    'rollout_avg_horizon': {
        'label': 'Horizon Steps',
        'title': 'Rollout Average Horizon',
        'fname': 'rollout_avg_horizon.png',
    },
}


def save_figure(fig, path: str):
    ax_list = fig.get_axes()
    assert len(ax_list) == 1, (
        f"Figure must have exactly 1 Axes (got {len(ax_list)}). "
        "Each metric must be saved to a separate file."
    )
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def load_seed_csv(csv_path: str) -> dict:
    """seed_N_progress.csv → {col: list[float]}"""
    data = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k is None:
                    continue
                try:
                    val = float(v) if v not in ('nan', '', 'None') else float('nan')
                except (ValueError, TypeError):
                    val = float('nan')
                data[k].append(val)
    return dict(data)


def load_all_seeds(results_dir: str) -> dict:
    """results_dir 내 seed_N_progress.csv 파일을 모두 로드 → {seed_id: {col: list}}"""
    all_data = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('_progress.csv'):
            continue
        # seed_N_progress.csv → N
        try:
            seed_id = int(fname.split('_')[1])
        except (IndexError, ValueError):
            continue
        csv_path = os.path.join(results_dir, fname)
        all_data[seed_id] = load_seed_csv(csv_path)
        print(f'Loaded seed {seed_id}: {csv_path}')
    return all_data


def compute_mean_std(all_data: dict, col: str):
    """에피소드 인덱스 기준으로 시드 정렬 후 mean±std 계산."""
    seeds = sorted(all_data.keys())
    lengths = [len(all_data[s].get(col, [])) for s in seeds]
    max_ep = max(lengths) if lengths else 0
    if max_ep == 0:
        return np.array([]), np.array([]), np.array([])
    matrix = np.full((len(seeds), max_ep), np.nan)
    for i, s in enumerate(seeds):
        vals = all_data[s].get(col, [])
        matrix[i, :len(vals)] = vals
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    episodes = np.arange(1, max_ep + 1)
    return episodes, mean, std


def plot_per_seed(all_data: dict, figures_dir: str):
    """시드별 개별 figure → figures_dir/seed_N/ 하위에 저장."""
    for seed_id, seed_data in sorted(all_data.items()):
        seed_dir = os.path.join(figures_dir, f'seed_{seed_id}')
        os.makedirs(seed_dir, exist_ok=True)

        episodes = np.arange(1, len(next(iter(seed_data.values()))) + 1)

        for col, meta in METRIC_META.items():
            vals = seed_data.get(col)
            if vals is None:
                continue
            arr = np.array(vals, dtype=float)
            valid = ~np.isnan(arr)
            if not valid.any():
                continue

            ep = np.arange(1, len(arr) + 1)
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
            ax.plot(ep[valid], arr[valid], color='steelblue', linewidth=1.5)
            if 'hline' in meta:
                ax.axhline(y=meta['hline'], color='red', linestyle='--',
                           linewidth=1.5, label=str(meta['hline']))
                ax.legend()
            ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax.set_ylabel(meta['label'], fontsize=12, fontweight='bold')
            ax.set_title(f'{meta["title"]} — Seed {seed_id}', fontsize=13, fontweight='bold')
            ax.grid(True)
            save_figure(fig, os.path.join(seed_dir, meta['fname']))

        print(f'  Seed {seed_id} figures saved to {seed_dir}')


def smooth(arr: np.ndarray, window: int = 11) -> tuple:
    """window 크기 uniform moving average. mode='valid'로 edge 왜곡 없음.
    반환: (trim, smoothed) — trim은 양끝에서 잘라낼 인덱스 수."""
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode='valid')
    trim = (window - 1) // 2  # 양끝에서 잘라낼 수 (window=11 → 5)
    return trim, smoothed


def plot_multiseed(all_data: dict, figures_dir: str):
    """8시드 mean±std figure → figures_dir/multiseed/ 하위에 저장."""
    multiseed_dir = os.path.join(figures_dir, 'multiseed')
    os.makedirs(multiseed_dir, exist_ok=True)
    num_seeds = len(all_data)

    for col, meta in METRIC_META.items():
        episodes, mean, std = compute_mean_std(all_data, col)
        if len(episodes) == 0:
            continue
        valid = ~np.isnan(mean)
        if not valid.any():
            print(f'Skip {col}: all NaN')
            continue

        ep_v, mean_v, std_v = episodes[valid], mean[valid], std[valid]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
        ax.fill_between(ep_v, mean_v - std_v, mean_v + std_v,
                        color='steelblue', alpha=0.2, linewidth=0, label='±1 Std')
        ax.plot(ep_v, mean_v, color='steelblue', linewidth=2.0, label='Mean')
        if 'hline' in meta:
            ax.axhline(y=meta['hline'], color='red', linestyle='--',
                       linewidth=1.5, label=str(meta['hline']))
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel(meta['label'], fontsize=12, fontweight='bold')
        ax.set_title(f'{meta["title"]} — Mean ± Std ({num_seeds} seeds)',
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True)
        save_figure(fig, os.path.join(multiseed_dir, meta['fname']))

    print(f'Multi-seed figures saved to {multiseed_dir}')


def main():
    parser = argparse.ArgumentParser(description='Plot real_ratio experiment results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='seed_N_progress.csv 파일들이 있는 디렉토리')
    parser.add_argument('--figures-dir', type=str, default=None,
                        help='figure 저장 디렉토리 (기본: Cycle_1-dob-rl/figures/real_ratio=1.0)')
    args = parser.parse_args()

    if args.figures_dir is None:
        cycle_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ratio_name = os.path.basename(args.results_dir)
        args.figures_dir = os.path.join(cycle_root, 'figures', ratio_name)

    os.makedirs(args.figures_dir, exist_ok=True)

    all_data = load_all_seeds(args.results_dir)
    if not all_data:
        print(f'No seed_N_progress.csv files found in {args.results_dir}')
        return

    print(f'\n=== Per-seed figures ===')
    plot_per_seed(all_data, args.figures_dir)

    print(f'\n=== Multi-seed mean±std figures ===')
    plot_multiseed(all_data, args.figures_dir)

    print(f'\nDone. Figures saved to: {args.figures_dir}')


if __name__ == '__main__':
    main()
