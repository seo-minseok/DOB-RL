"""
preview_uncertainty_sampling.py — 더미 데이터로 uncertainty 시각화 미리보기 (mean ± std)

Usage:
  python scripts/preview_uncertainty_sampling.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def smooth(arr, window=10):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def save_figure(fig, path: str):
    axes = fig.get_axes()
    assert len(axes) == 1, f"1 figure = 1 Axes 규칙 위반: {len(axes)}개 Axes 감지"
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')


def generate_dummy_data(num_episodes=200, warm_start=5, num_seeds=5):
    """멀티시드 더미 uncertainty 데이터 생성."""
    episodes = np.arange(1, num_episodes + 1)
    buf_all  = np.full((num_seeds, num_episodes), np.nan)
    samp_all = np.full((num_seeds, num_episodes), np.nan)

    active = np.arange(warm_start, num_episodes)
    t      = np.linspace(0, 1, len(active))

    for s in range(num_seeds):
        rng = np.random.default_rng(s)
        buf_base  = 0.8 * np.exp(-3.5 * t) + 0.15
        buf_noise = rng.normal(0, 0.04, len(active))
        buf_all[s, active] = np.clip(buf_base + buf_noise, 0.05, None)

        ratio      = 1.5 - 0.4 * t
        samp_noise = rng.normal(0, 0.05, len(active))
        samp_all[s, active] = np.clip(buf_all[s, active] * ratio + samp_noise, 0.05, None)

    return episodes, buf_all, samp_all


def plot_mean_std(ax, episodes, data_all, color, label, window=10):
    """
    data_all: (num_seeds, num_episodes) — NaN은 model training 미수행 에피소드
    유효 에피소드만 집계 후 mean ± std를 이동 평균으로 스무딩.
    """
    valid_mask = ~np.all(np.isnan(data_all), axis=0)
    ep_valid   = episodes[valid_mask]
    data_valid = data_all[:, valid_mask]  # (num_seeds, T)

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


def main():
    num_seeds = 5
    episodes, buf_all, samp_all = generate_dummy_data(num_seeds=num_seeds)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    plot_mean_std(ax, episodes, buf_all,  color='steelblue',
                  label=f'Buffer — all data (mean ± std, {num_seeds} seeds)')
    plot_mean_std(ax, episodes, samp_all, color='darkorange',
                  label=f'Sampled — model train (mean ± std, {num_seeds} seeds)')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Uncertainty Magnitude (L2 norm)')
    ax.set_title('Buffer vs Sampled Uncertainty over Training  [DUMMY DATA]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.normpath(
        os.path.join(script_dir, '..', 'figures', 'preview_uncertainty_sampling.png')
    )
    save_figure(fig, out_path)
    plt.close(fig)


if __name__ == '__main__':
    main()
