"""
plot_csv_metrics.py — seed_N_progress.csv → per-metric figures

Usage:
  python scripts/plot_csv_metrics.py --results-dir ./results/default \
                                      --figures-dir ./results/default/figures
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# x축으로 쓸 컬럼 및 항상 sparse로 처리할 컬럼
_X_COLS    = {'episode', 'total_steps', 'ep_steps'}
_SPARSE    = {'eval_reward_avg'}


def detect_metrics(dfs: dict) -> list:
    """CSV 컬럼을 자동 감지해서 (col, ylabel, sparse) 리스트 반환."""
    import re
    sample = next(iter(dfs.values()))
    metrics = []
    for col in sample.columns:
        if col in _X_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(sample[col]):
            continue
        # pandas 중복 열 자동 생성 이름 (예: eval_reward_avg.1) 제외
        if re.search(r'\.\d+$', col):
            continue
        sparse = col in _SPARSE
        ylabel = col.replace('_', ' ').title()
        metrics.append((col, ylabel, sparse))
    return metrics

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


# ── 데이터 로딩 ────────────────────────────────────────────────────────────
def load_csvs(results_dir: str) -> dict[int, pd.DataFrame]:
    dfs = {}
    for seed in range(1, 9):
        path = os.path.join(results_dir, f'seed_{seed}_progress.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        # 중복 열 이름 처리 (eval_reward_avg 두 번 등장)
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


# ── 공통 x 그리드 보간 ─────────────────────────────────────────────────────
def interpolate_to_grid(xs: np.ndarray, ys: np.ndarray,
                        grid: np.ndarray) -> np.ndarray:
    mask = np.isfinite(ys)
    if mask.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, xs[mask], ys[mask])


def build_common_grid(dfs: dict, n_points: int = 500) -> np.ndarray:
    x_min = min(df['total_steps'].min() for df in dfs.values())
    x_max = max(df['total_steps'].max() for df in dfs.values())
    return np.linspace(x_min, x_max, n_points)


# ── per-seed 개별 figure ───────────────────────────────────────────────────
def plot_per_seed(dfs: dict, col: str, ylabel: str, sparse: bool,
                  out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for seed, df in dfs.items():
        if col not in df.columns:
            continue
        xs = df['total_steps'].values
        ys = df[col].values.astype(float)

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        if sparse:
            # eval_reward_avg: scatter + line on non-NaN points only
            mask = np.isfinite(ys)
            ax.plot(xs[mask], ys[mask], marker='o', ms=3,
                    color=COLORS[(seed - 1) % len(COLORS)],
                    label=f'seed {seed}')
        else:
            ax.plot(xs, ys, lw=0.8, alpha=0.85,
                    color=COLORS[(seed - 1) % len(COLORS)],
                    label=f'seed {seed}')

        ax.set_xlabel('Total Steps')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Seed {seed} — {ylabel}')
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(out_dir, f'seed_{seed}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')


# ── mean ± std figure ──────────────────────────────────────────────────────
def plot_mean_std(dfs: dict, col: str, ylabel: str, sparse: bool,
                  grid: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    if sparse:
        # eval_reward_avg: step 기준으로 non-NaN 포인트만 수집 후 보간
        mat = np.stack([
            interpolate_to_grid(
                df['total_steps'].values,
                df[col].values.astype(float),
                grid,
            )
            for df in dfs.values() if col in df.columns
        ])
    else:
        mat = np.stack([
            interpolate_to_grid(
                df['total_steps'].values,
                df[col].values.astype(float),
                grid,
            )
            for df in dfs.values() if col in df.columns
        ])

    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(grid, mean, lw=1.5, color='steelblue', label='mean')
    ax.fill_between(grid, mean - std, mean + std,
                    alpha=0.25, color='steelblue', label='mean ± std (8 seeds)')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Mean ± Std (8 seeds) — {ylabel}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(out_dir, f'{col}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── all-seeds overlay figure ───────────────────────────────────────────────
def plot_all_overlay(dfs: dict, col: str, ylabel: str, sparse: bool,
                     out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for seed, df in dfs.items():
        if col not in df.columns:
            continue
        xs = df['total_steps'].values
        ys = df[col].values.astype(float)
        color = COLORS[(seed - 1) % len(COLORS)]
        if sparse:
            mask = np.isfinite(ys)
            ax.plot(xs[mask], ys[mask], marker='o', ms=2.5, lw=0.8,
                    color=color, alpha=0.75, label=f'seed {seed}')
        else:
            ax.plot(xs, ys, lw=0.7, alpha=0.65,
                    color=color, label=f'seed {seed}')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel(ylabel)
    ax.set_title(f'All Seeds — {ylabel}')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(out_dir, f'{col}_all_seeds.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ── main ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='./results/default')
    p.add_argument('--figures-dir', default='./results/default/figures')
    return p.parse_args()


def main():
    args = parse_args()
    dfs  = load_csvs(args.results_dir)
    if not dfs:
        print(f'No seed CSVs found in {args.results_dir}')
        return
    print(f'Loaded {len(dfs)} seeds: {sorted(dfs.keys())}')

    grid    = build_common_grid(dfs)
    METRICS = detect_metrics(dfs)

    for col, ylabel, sparse in METRICS:
        if not any(col in df.columns for df in dfs.values()):
            print(f'  Skipping {col} (not found)')
            continue

        print(f'\n[{col}]')

        # 1) per-seed: figures/default/seed_N/{metric}.png
        for seed, df in dfs.items():
            if col not in df.columns:
                continue
            seed_dir = os.path.join(args.figures_dir, f'seed_{seed}')
            os.makedirs(seed_dir, exist_ok=True)

            xs = df['total_steps'].values
            ys = df[col].values.astype(float)

            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            if sparse:
                mask = np.isfinite(ys)
                ax.plot(xs[mask], ys[mask], marker='o', ms=3,
                        color=COLORS[(seed - 1) % len(COLORS)],
                        label=f'seed {seed}')
            else:
                ax.plot(xs, ys, lw=0.8, alpha=0.85,
                        color=COLORS[(seed - 1) % len(COLORS)],
                        label=f'seed {seed}')
            ax.set_xlabel('Total Steps')
            ax.set_ylabel(ylabel)
            ax.set_title(f'Seed {seed} — {ylabel}')
            ax.legend(fontsize=8)
            fig.tight_layout()
            out = os.path.join(seed_dir, f'{col}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {out}')

        # 2) mean±std: figures/default/mean_std/{metric}.png
        mean_std_dir = os.path.join(args.figures_dir, 'mean_std')
        plot_mean_std(dfs, col, ylabel, sparse, grid, mean_std_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
