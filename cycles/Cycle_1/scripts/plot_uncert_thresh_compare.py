"""
plot_uncert_thresh_compare.py — uncert_thresh 4개 조건 mean±std 오버레이 비교

Usage:
  python scripts/plot_uncert_thresh_compare.py \
    --results-root ./results \
    --real-ratio   0.05 \
    --thresholds   0.5 1.0 1.5 2.0 \
    --figures-dir  ./figures/compare_uncert_thresh
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
_COLORS = ['steelblue', 'orange', 'green', 'red']


def load_seeds(results_dir: str) -> dict[str, pd.DataFrame]:
    dfs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'seed_*_progress.csv'))):
        seed = os.path.basename(path).split('_')[1]
        df = pd.read_csv(path, header=0)
        cols = list(df.columns)
        seen: dict = {}
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


def interp_mean_std(dfs: dict, metric: str, n_points: int = 500):
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
                 datasets: list,
                 x_max: float,
                 out_path: str):
    series = []
    for dfs, label, color in datasets:
        grid, mean, std = interp_mean_std(dfs, metric)
        if grid is not None:
            series.append((grid, mean, std, label, color))

    if not series:
        return

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    for grid, mean, std, label, color in series:
        mask = grid <= x_max
        ax.plot(grid[mask], mean[mask], color=color, label=label, linewidth=1.5)
        ax.fill_between(grid[mask], (mean - std)[mask], (mean + std)[mask],
                        color=color, alpha=0.2)

    n_seeds_str = ', '.join(f'{label}:{len(dfs)}' for dfs, label, _ in datasets)
    ylabel = metric.replace('_', ' ').title()
    ax.set_xlabel('Total Steps')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} — mean ± std  ({n_seeds_str})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=str, required=True)
    parser.add_argument('--real-ratio',   type=str, default='0.05')
    parser.add_argument('--thresholds',   type=float, nargs='+',
                        default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument('--figures-dir',  type=str, required=True)
    parser.add_argument('--x-max',        type=float, default=None)
    args = parser.parse_args()

    datasets = []
    for thresh, color in zip(args.thresholds, _COLORS):
        subdir = os.path.join(
            args.results_root,
            f'real_ratio={args.real_ratio}-uncert_thresh={thresh}'
        )
        dfs = load_seeds(subdir)
        if not dfs:
            print(f'[WARN] No CSVs in {subdir}, skipping')
            continue
        label = f'thresh={thresh}'
        datasets.append((dfs, label, color))
        print(f'{label}: {len(dfs)} seeds')

    if not datasets:
        print('No data loaded.'); return

    # x_max: 각 조건의 seed별 최솟값 중 최솟값 (공정 비교)
    if args.x_max is not None:
        x_max = args.x_max
    else:
        x_max = min(
            min(df[_X_COL].max() for df in dfs.values())
            for dfs, _, _ in datasets
        )
    print(f'x_max = {x_max:.0f}')

    # 공통 메트릭 (모든 조건에 존재하는 컬럼)
    skip = {'episode', 'total_steps', 'ep_steps', 'episode_length'}
    all_metric_sets = []
    for dfs, _, _ in datasets:
        sample = next(iter(dfs.values()))
        cols = {c for c in sample.columns
                if c not in skip and pd.api.types.is_numeric_dtype(sample[c])}
        all_metric_sets.append(cols)
    metrics = sorted(set.intersection(*all_metric_sets))
    print(f'Common metrics: {metrics}')

    os.makedirs(args.figures_dir, exist_ok=True)
    for m in metrics:
        out = os.path.join(args.figures_dir, f'{m}.png')
        plot_overlay(m, datasets, x_max, out)

    print('Done.')


if __name__ == '__main__':
    main()
