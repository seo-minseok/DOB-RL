"""
plot_mbpo_vs_dob.py — MBPO vs DOB-MBRL mean±std 오버레이 비교

Usage:
  python scripts/plot_mbpo_vs_dob.py \
    --results-mbpo  ./results/mbpo \
    --results-dob   ./results/real_ratio=0.05-uncert_thresh=2.0 \
    --figures-dir   ./figures/compare_mbpo_vs_dob-mbrl_uncert2.0
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_X_COL = 'total_steps'

# 공통 컬럼 (같은 이름으로 양쪽에 존재)
_COMMON_METRICS = ['env_return', 'eval_reward_avg', 'reward', 'td_loss_avg']

# 컬럼명이 다른 커스텀 매핑: (출력파일명, mbpo_col, dob_col, ylabel)
_CUSTOM_METRICS = [
    ('model_error',          'model_error_avg',    'residual_error_avg', 'Model Error'),
    ('rollout_length_compare', 'rollout_mean_length', 'rollout_avg_horizon', 'Rollout Length'),
]


def load_seeds(results_dir: str) -> dict[str, pd.DataFrame]:
    dfs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'seed_*_progress.csv'))):
        seed = os.path.basename(path).split('_')[1]
        df = pd.read_csv(path, header=0)
        # 중복 컬럼 처리 (eval_reward_avg 등)
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
    all_steps, all_vals = [], []
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

    return grid, np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def save_overlay(out_path: str,
                 series: list,   # [(grid, mean, std, label, color), ...]
                 x_max: float,
                 ylabel: str,
                 title: str):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    for grid, mean, std, label, color in series:
        mask = grid <= x_max
        ax.plot(grid[mask], mean[mask], color=color, label=label, linewidth=1.5)
        ax.fill_between(grid[mask], (mean - std)[mask], (mean + std)[mask],
                        color=color, alpha=0.2)
    ax.set_xlabel('Total Steps')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-mbpo', type=str, required=True)
    parser.add_argument('--results-dob',  type=str, required=True)
    parser.add_argument('--figures-dir',  type=str, required=True)
    parser.add_argument('--x-max',        type=float, default=None)
    args = parser.parse_args()

    dfs_mbpo = load_seeds(args.results_mbpo)
    dfs_dob  = load_seeds(args.results_dob)

    if not dfs_mbpo:
        print(f'No CSVs in {args.results_mbpo}'); return
    if not dfs_dob:
        print(f'No CSVs in {args.results_dob}'); return

    print(f'MBPO: {len(dfs_mbpo)} seeds')
    print(f'DOB-MBRL: {len(dfs_dob)} seeds')

    if args.x_max is not None:
        x_max = args.x_max
    else:
        x_max = min(
            min(df[_X_COL].max() for df in dfs_mbpo.values()),
            min(df[_X_COL].max() for df in dfs_dob.values()),
        )
    print(f'x_max = {x_max:.0f}')

    os.makedirs(args.figures_dir, exist_ok=True)

    # 공통 메트릭
    for metric in _COMMON_METRICS:
        g_m, mean_m, std_m = interp_mean_std(dfs_mbpo, metric)
        g_d, mean_d, std_d = interp_mean_std(dfs_dob,  metric)
        series = []
        if g_m is not None:
            series.append((g_m, mean_m, std_m, 'MBPO', 'orange'))
        if g_d is not None:
            series.append((g_d, mean_d, std_d, 'DOB-MBRL', 'steelblue'))
        if not series:
            continue
        ylabel = metric.replace('_', ' ').title()
        n_str = f'MBPO:{len(dfs_mbpo)}, DOB-MBRL:{len(dfs_dob)}'
        save_overlay(
            out_path=os.path.join(args.figures_dir, f'{metric}.png'),
            series=series,
            x_max=x_max,
            ylabel=ylabel,
            title=f'{ylabel} — mean ± std  ({n_str})',
        )

    # 커스텀 매핑 메트릭
    for fname, mbpo_col, dob_col, ylabel in _CUSTOM_METRICS:
        g_m, mean_m, std_m = interp_mean_std(dfs_mbpo, mbpo_col)
        g_d, mean_d, std_d = interp_mean_std(dfs_dob,  dob_col)
        series = []
        if g_m is not None:
            series.append((g_m, mean_m, std_m, f'MBPO ({mbpo_col})', 'orange'))
        if g_d is not None:
            series.append((g_d, mean_d, std_d, f'DOB-MBRL ({dob_col})', 'steelblue'))
        if not series:
            continue
        n_str = f'MBPO:{len(dfs_mbpo)}, DOB-MBRL:{len(dfs_dob)}'
        save_overlay(
            out_path=os.path.join(args.figures_dir, f'{fname}.png'),
            series=series,
            x_max=x_max,
            ylabel=ylabel,
            title=f'{ylabel} — mean ± std  ({n_str})',
        )

    print('Done.')


if __name__ == '__main__':
    main()
