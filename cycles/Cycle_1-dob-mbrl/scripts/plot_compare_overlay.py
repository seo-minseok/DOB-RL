"""
plot_compare_overlay.py — Cycle_1-dob-rl vs Cycle_1-mbpo mean±std 오버레이 비교

Usage:
  python scripts/plot_compare_overlay.py \
    --results-a  ./results/real_ratio=0.05 \
    --label-a    "Cycle_1-dob-rl" \
    --results-b  ../Cycle_1-mbpo/results/default \
    --label-b    "Cycle_1-mbpo" \
    --color-b    orange \
    --figures-dir ./figures/compare_adam_vs_mbpo
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_X_COL  = 'total_steps'
_SPARSE = {'eval_reward_avg'}


def load_seeds(results_dir: str) -> dict[str, pd.DataFrame]:
    """seed_N_progress.csv 파일들을 로드해 {seed: df} 반환."""
    dfs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'seed_*_progress.csv'))):
        seed = os.path.basename(path).split('_')[1]
        df = pd.read_csv(path, header=0)
        # 중복 컬럼 처리 (eval_reward_avg 등)
        cols = list(df.columns)
        seen = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_cols.append(f'{c}_{seen[c]}')
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols
        dfs[seed] = df
    return dfs


def common_metrics(dfs_a: dict, dfs_b: dict) -> list[str]:
    """양쪽에 존재하는 수치형 메트릭 컬럼 반환 (x축 컬럼 제외)."""
    _skip = {'episode', 'total_steps', 'ep_steps', 'episode_length'}
    sample_a = next(iter(dfs_a.values()))
    sample_b = next(iter(dfs_b.values()))
    cols_a = set(c for c in sample_a.columns
                 if c not in _skip and pd.api.types.is_numeric_dtype(sample_a[c]))
    cols_b = set(c for c in sample_b.columns
                 if c not in _skip and pd.api.types.is_numeric_dtype(sample_b[c]))
    return sorted(cols_a & cols_b)


def interp_mean_std(dfs: dict, metric: str, n_points: int = 500):
    """공통 grid에서 mean, std 계산. sparse 메트릭은 ffill 없이 그대로."""
    all_steps = []
    all_vals  = []
    for df in dfs.values():
        if metric not in df.columns:
            continue
        sub = df[[_X_COL, metric]].dropna()
        if sub.empty:
            continue
        all_steps.append(sub[_X_COL].values)
        all_vals.append(sub[metric].values)

    if not all_steps:
        return None, None, None

    x_min = max(s.min() for s in all_steps)
    x_max = min(s.max() for s in all_steps)
    if x_min >= x_max:
        return None, None, None

    grid = np.linspace(x_min, x_max, n_points)
    mat  = np.full((len(all_steps), n_points), np.nan)
    for i, (xs, ys) in enumerate(zip(all_steps, all_vals)):
        mat[i] = np.interp(grid, xs, ys)

    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)
    return grid, mean, std


def plot_overlay(metric: str,
                 datasets: list,   # list of (dfs, label, color)
                 x_max: float,
                 out_path: str):
    series = []
    for dfs, label, color in datasets:
        grid, mean, std = interp_mean_std(dfs, metric)
        if grid is not None:
            series.append((grid, mean, std, label, color))

    if not series:
        return

    # x축 상한: 지정된 x_max 이하로 클리핑
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for grid, mean, std, label, color in series:
        mask = grid <= x_max
        ax.plot(grid[mask], mean[mask], color=color, label=label, linewidth=1.5)
        ax.fill_between(grid[mask], (mean - std)[mask], (mean + std)[mask],
                        color=color, alpha=0.2)

    ylabel = metric.replace('_', ' ').title()
    ax.set_xlabel('Total Steps')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} — mean ± std (8 seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-a',   type=str, required=True)
    parser.add_argument('--label-a',     type=str, default='A')
    parser.add_argument('--color-a',     type=str, default='steelblue')
    parser.add_argument('--results-b',   type=str, required=True)
    parser.add_argument('--label-b',     type=str, default='B')
    parser.add_argument('--color-b',     type=str, default='orange')
    parser.add_argument('--results-c',   type=str, default=None)
    parser.add_argument('--label-c',     type=str, default='C')
    parser.add_argument('--color-c',     type=str, default='green')
    parser.add_argument('--figures-dir', type=str, required=True)
    parser.add_argument('--x-max',       type=float, default=None,
                        help='x축 상한 (미지정 시 전체 범위)')
    args = parser.parse_args()

    dfs_a = load_seeds(args.results_a)
    dfs_b = load_seeds(args.results_b)

    if not dfs_a:
        print(f'No CSVs found in {args.results_a}'); return
    if not dfs_b:
        print(f'No CSVs found in {args.results_b}'); return

    datasets = [
        (dfs_a, args.label_a, args.color_a),
        (dfs_b, args.label_b, args.color_b),
    ]

    if args.results_c:
        dfs_c = load_seeds(args.results_c)
        if dfs_c:
            datasets.append((dfs_c, args.label_c, args.color_c))

    # 공통 메트릭: 모든 데이터셋에 걸쳐 교집합
    all_dfs = [d for d, _, _ in datasets]
    metrics = common_metrics(all_dfs[0], all_dfs[1])
    for extra in all_dfs[2:]:
        metrics = [m for m in metrics if m in next(iter(extra.values())).columns]

    # x_max: 미지정 시 a, b의 최솟값 (c는 더 길 수 있으므로)
    if args.x_max is not None:
        x_max = args.x_max
    else:
        x_max = min(
            next(iter(dfs_a.values()))['total_steps'].max(),
            next(iter(dfs_b.values()))['total_steps'].max(),
        )

    print(f'Common metrics: {metrics}  x_max={x_max:.0f}')

    os.makedirs(args.figures_dir, exist_ok=True)
    for m in metrics:
        out = os.path.join(args.figures_dir, f'{m}.png')
        plot_overlay(m, datasets, x_max, out)

    print('Done.')


if __name__ == '__main__':
    main()
