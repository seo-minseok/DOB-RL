"""
plot_multiseed_metrics.py
8개 시드의 각 metric을 mean±std로 그림. metric당 파일 1개.
Usage:
  python scripts/plot_multiseed_metrics.py --results-dir <path> --out-dir <path>
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

METRICS = [
    ('reward',              'Episode Reward (train)',       'Reward'),
    ('eval_reward_avg',     'Eval Reward (avg 10 ep)',      'Reward'),
    ('td_loss_avg',         'TD Loss',                      'Loss'),
    ('res_net_loss',        'Residual Net Loss',            'Loss'),
    ('rollout_pass_rate',   'Rollout Pass Rate',            'Rate'),
    ('rollout_avg_horizon', 'Rollout Avg Horizon',          'Steps'),
    ('nominal_error_avg',   'Nominal Model Error',          'Error'),
    ('residual_error_avg',  'Residual Model Error',         'Error'),
    ('dhat_norm_avg',       'DOB Estimate Norm',            'Norm'),
]

SMOOTH_WINDOW = 20   # reward 스무딩 윈도우 (에피소드 단위)
NUM_GRID_PTS  = 500  # 공통 x축 grid 포인트 수


def load_seed(path, metric):
    steps, vals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row[metric])
                if not np.isnan(v):
                    steps.append(int(row['total_steps']))
                    vals.append(v)
            except (ValueError, KeyError):
                pass
    return np.array(steps), np.array(vals)


def smooth(arr, w):
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='valid')


def save_figure(fig, path):
    axes = fig.get_axes()
    assert len(axes) == 1, f"subplot 금지: axes={len(axes)}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_metric(results_dir, out_dir, metric, title, ylabel, num_seeds=8):
    all_steps, all_vals = [], []
    for seed in range(1, num_seeds + 1):
        p = os.path.join(results_dir, f'seed_{seed}_progress.csv')
        if not os.path.exists(p):
            continue
        s, v = load_seed(p, metric)
        if len(s) < 2:
            continue
        if metric == 'reward':
            # 스무딩 적용 후 steps 조정
            v = smooth(v, SMOOTH_WINDOW)
            s = s[SMOOTH_WINDOW - 1:]
            if len(s) != len(v):
                s = s[:len(v)]
        all_steps.append(s)
        all_vals.append(v)

    if not all_steps:
        print(f"  Skipping {metric}: no data")
        return

    # 공통 x 그리드
    x_min = max(s[0]  for s in all_steps)
    x_max = min(s[-1] for s in all_steps)
    if x_min >= x_max:
        x_min = min(s[0]  for s in all_steps)
        x_max = max(s[-1] for s in all_steps)
    grid = np.linspace(x_min, x_max, NUM_GRID_PTS)

    # 각 시드를 grid에 interpolate
    interp_vals = []
    for s, v in zip(all_steps, all_vals):
        interp_vals.append(np.interp(grid, s, v))

    mat  = np.stack(interp_vals)   # (n_seeds, grid)
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(grid, mean, linewidth=1.8, label=f'Mean ({len(interp_vals)} seeds)')
    ax.fill_between(grid, mean - std, mean + std, alpha=0.25, label='±1 std')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}  [ngs=20, mh=15, rbs=100k, rr=0.1]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = metric + '.png'
    save_figure(fig, os.path.join(out_dir, fname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--out-dir',     required=True)
    args = parser.parse_args()

    print(f"results_dir : {args.results_dir}")
    print(f"out_dir     : {args.out_dir}")
    print()

    for metric, title, ylabel in METRICS:
        print(f"Plotting: {metric}")
        plot_metric(args.results_dir, args.out_dir, metric, title, ylabel)


if __name__ == '__main__':
    main()
