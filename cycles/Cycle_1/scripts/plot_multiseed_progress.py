"""
plot_multiseed_progress.py — per-seed CSV에서 멀티시드 mean±std 계산 후 figure 생성

Usage:
  python cycles/Cycle_1-dob-rl/scripts/plot_multiseed_progress.py \
      --results-dir cycles/Cycle_1-dob-rl/results/real_ratio=0.2 \
      --figures-dir cycles/Cycle_1-dob-rl/figures/real_ratio=0.2/multiseed
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_META = {
    'eval_reward_avg':    ('Eval Reward (avg)',       'Eval Reward (avg)'),
    'reward':             ('Cumulative Reward',        'Episode Reward'),
    'env_return':         ('Env Return',               'Env Return'),
    'ep_steps':           ('Steps',                    'Episode Length'),
    'nominal_error_avg':  ('Error',                    'Nominal Error (avg)'),
    'residual_error_avg': ('Error',                    'Residual Error (avg)'),
    'dhat_norm_avg':      ('||d_hat||',                'DOB Estimate Norm (avg)'),
    'uncertainty_avg':    ('Uncertainty',              'Uncertainty (avg)'),
    'res_net_loss':       ('Loss',                     'ResNet Loss'),
    'rbf_loss':           ('Loss',                     'RBF Loss'),
    'td_loss_avg':        ('TD Loss',                  'TD Loss (avg)'),
    'buffer_uncert_avg':  ('Uncertainty',              'Buffer Uncertainty (avg)'),
    'sampled_uncert_avg': ('Uncertainty',              'Sampled Uncertainty (avg)'),
    'fresh_uncert_avg':   ('Uncertainty',              'Fresh Uncertainty Target (avg)'),
    'rollout_uncert_avg': ('Uncertainty',              'Rollout Uncertainty (avg)'),
    'rollout_pass_rate':  ('Pass Rate',                'Rollout Pass Rate'),
    'rollout_avg_horizon': ('Horizon',                 'Rollout Avg Horizon'),
}


def load_all_seeds(results_dir: str):
    """results_dir 안의 seed_*_progress.csv를 모두 읽어 {seed: DataFrame} 반환."""
    pattern = os.path.join(results_dir, 'seed_*_progress.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No seed CSV files found in {results_dir}')

    seed_dfs = {}
    for path in files:
        basename = os.path.basename(path)
        seed_num = int(basename.split('_')[1])
        df = pd.read_csv(path)
        df = df.set_index('episode')
        seed_dfs[seed_num] = df
        print(f'  Seed {seed_num}: {len(df)} episodes')
    return seed_dfs


def compute_mean_std(seed_dfs: dict):
    """에피소드 인덱스 기준으로 정렬 후 mean, std 계산. 공통 에피소드 범위 내에서만."""
    all_episodes = sorted(set().union(*[set(df.index) for df in seed_dfs.values()]))
    all_cols = list(next(iter(seed_dfs.values())).columns)

    # 모든 시드를 동일한 에피소드 인덱스로 reindex (없는 에피소드는 NaN)
    aligned = {s: df.reindex(all_episodes) for s, df in seed_dfs.items()}
    stacked = np.stack([df.values for df in aligned.values()], axis=0)  # (S, E, C)

    mean_vals = np.nanmean(stacked, axis=0)
    std_vals  = np.nanstd(stacked, axis=0)

    mean_df = pd.DataFrame(mean_vals, index=all_episodes, columns=all_cols)
    std_df  = pd.DataFrame(std_vals,  index=all_episodes, columns=all_cols)
    mean_df.index.name = 'episode'
    std_df.index.name  = 'episode'
    return mean_df, std_df


def save_summary_csv(mean_df: pd.DataFrame, std_df: pd.DataFrame, out_dir: str):
    """mean과 std를 하나의 CSV로 저장 (컬럼명: metric_mean, metric_std)."""
    os.makedirs(out_dir, exist_ok=True)
    combined = pd.concat([
        mean_df.add_suffix('_mean'),
        std_df.add_suffix('_std'),
    ], axis=1)
    combined.sort_index(axis=1, inplace=True)
    out_path = os.path.join(out_dir, 'multiseed_summary.csv')
    combined.to_csv(out_path)
    print(f'Saved summary CSV: {out_path}')
    return out_path


def save_figure(fig, path: str):
    ax_list = fig.get_axes()
    assert len(ax_list) == 1, (
        f"Figure must have exactly 1 Axes (got {len(ax_list)}). "
        "Each metric must be saved to a separate file."
    )
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {path}')


def plot_all_metrics(mean_df: pd.DataFrame, std_df: pd.DataFrame,
                     num_seeds: int, figures_dir: str, real_ratio: str):
    os.makedirs(figures_dir, exist_ok=True)
    episodes = mean_df.index.to_numpy()

    for col, (ylabel, title) in METRIC_META.items():
        if col not in mean_df.columns:
            continue
        mean_arr = mean_df[col].to_numpy()
        std_arr  = std_df[col].to_numpy()

        valid = ~np.isnan(mean_arr)
        if not valid.any():
            print(f'Skip {col}: all NaN')
            continue

        ep_v, mean_v, std_v = episodes[valid], mean_arr[valid], std_arr[valid]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor='white')
        ax.fill_between(ep_v, mean_v - std_v, mean_v + std_v,
                        color='steelblue', alpha=0.2, linewidth=0, label='±1 Std')
        ax.plot(ep_v, mean_v, color='steelblue', linewidth=2.0, label='Mean')
        if col in ('reward', 'eval_reward_avg', 'env_return'):
            ax.axhline(y=480, color='red', linestyle='--', linewidth=1.5, label='Target (480)')
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(
            f'{title} — Mean ± Std ({num_seeds} seeds, real_ratio={real_ratio})',
            fontsize=13, fontweight='bold'
        )
        ax.legend()
        ax.grid(True)
        save_figure(fig, os.path.join(figures_dir, f'{col}.png'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True,
                        help='seed_*_progress.csv 파일이 있는 디렉토리')
    parser.add_argument('--figures-dir', default=None,
                        help='figure 저장 디렉토리 (기본: results-dir/../figures/...)')
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir.rstrip('/')
    real_ratio  = os.path.basename(results_dir).replace('real_ratio=', '')

    figures_dir = args.figures_dir or os.path.join(
        os.path.dirname(os.path.dirname(results_dir)),
        'figures',
        os.path.basename(results_dir),
        'multiseed',
    )

    print(f'Results dir : {results_dir}')
    print(f'Figures dir : {figures_dir}')

    print('Loading seed CSVs...')
    seed_dfs = load_all_seeds(results_dir)
    num_seeds = len(seed_dfs)

    print('Computing mean ± std...')
    mean_df, std_df = compute_mean_std(seed_dfs)

    save_summary_csv(mean_df, std_df, figures_dir)
    plot_all_metrics(mean_df, std_df, num_seeds, figures_dir, real_ratio)

    print(f'\nDone. {len(METRIC_META)} metrics processed.')


if __name__ == '__main__':
    main()
