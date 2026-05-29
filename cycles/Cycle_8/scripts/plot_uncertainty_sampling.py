"""
plot_uncertainty_sampling.py — 버퍼 전체 vs 샘플링된 데이터의 uncertainty 시각화 (mean ± std)

단일 시드:
  python scripts/plot_uncertainty_sampling.py --csv logs/seed_1_result.csv

멀티 시드 (디렉토리 내 seed_*.csv 자동 수집):
  python scripts/plot_uncertainty_sampling.py --csv-dir logs/
  python scripts/plot_uncertainty_sampling.py --csv-dir logs/ --out figures/uncertainty_sampling.png
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_figure(fig, path: str):
    axes = fig.get_axes()
    assert len(axes) == 1, f"1 figure = 1 Axes 규칙 위반: {len(axes)}개 Axes 감지"
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')


def smooth(arr, window=10):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def load_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    통합 CSV (seed 컬럼 포함) 또는 단일 시드 CSV를 로드해
    (episodes, buf_all, samp_all) 반환.
    buf/samp_all shape: (num_seeds, num_episodes)
    """
    df = pd.read_csv(csv_path)
    required = {'episode', 'buffer_uncert_avg', 'sampled_uncert_avg'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"컬럼 누락 {missing}\n"
            "model_learning.py / trainer.py / run_multi_seed.py 수정 후 재학습 필요."
        )

    if 'seed' in df.columns:
        seeds    = sorted(df['seed'].unique())
        num_ep   = df['episode'].max()
        episodes = np.arange(1, num_ep + 1)
        buf_all  = np.full((len(seeds), num_ep), np.nan)
        samp_all = np.full((len(seeds), num_ep), np.nan)
        for i, s in enumerate(seeds):
            sub    = df[df['seed'] == s]
            ep_idx = sub['episode'].to_numpy(dtype=int) - 1
            buf_all[i,  ep_idx] = sub['buffer_uncert_avg'].to_numpy(dtype=float)
            samp_all[i, ep_idx] = sub['sampled_uncert_avg'].to_numpy(dtype=float)
    else:
        num_ep   = len(df)
        episodes = np.arange(1, num_ep + 1)
        buf_all  = df['buffer_uncert_avg'].to_numpy(dtype=float).reshape(1, -1)
        samp_all = df['sampled_uncert_avg'].to_numpy(dtype=float).reshape(1, -1)

    return episodes, buf_all, samp_all


def plot_mean_std(ax, episodes, data_all, color, label, window=10):
    valid_mask = ~np.all(np.isnan(data_all), axis=0)
    ep_valid   = episodes[valid_mask]
    data_valid = data_all[:, valid_mask]

    mean = np.nanmean(data_valid, axis=0)
    std  = np.nanstd(data_valid,  axis=0)

    w = window
    if len(ep_valid) >= w:
        ep_sm   = ep_valid[w - 1:]
        mean_sm = smooth(mean, w)
        std_sm  = smooth(std,  w)
    else:
        ep_sm, mean_sm, std_sm = ep_valid, mean, std

    ax.plot(ep_sm, mean_sm, color=color, linewidth=2.0, label=label)
    ax.fill_between(ep_sm, mean_sm - std_sm, mean_sm + std_sm,
                    color=color, alpha=0.2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',    type=str, required=True,
                        help='통합 CSV 경로 (seed 컬럼 포함) 또는 단일 시드 CSV')
    parser.add_argument('--out',    type=str, default=None,
                        help='저장 경로 (기본: figures/baseline/uncertainty_sampling.png)')
    parser.add_argument('--smooth', type=int, default=10,
                        help='이동 평균 윈도우 (기본 10)')
    return parser.parse_args()


def main():
    args = parse_args()

    episodes, buf_all, samp_all = load_csv(args.csv)
    num_seeds = buf_all.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    plot_mean_std(ax, episodes, buf_all,  color='steelblue',
                  label=f'Buffer — all data (mean ± std, {num_seeds} seeds)',
                  window=args.smooth)
    plot_mean_std(ax, episodes, samp_all, color='darkorange',
                  label=f'Sampled — model train (mean ± std, {num_seeds} seeds)',
                  window=args.smooth)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Uncertainty Magnitude (L2 norm)')
    ax.set_title('Buffer vs Sampled Uncertainty over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if args.out:
        out_path = args.out
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path   = os.path.normpath(
            os.path.join(script_dir, '..', 'figures', 'baseline', 'uncertainty_sampling.png')
        )

    save_figure(fig, out_path)
    plt.close(fig)


if __name__ == '__main__':
    main()
