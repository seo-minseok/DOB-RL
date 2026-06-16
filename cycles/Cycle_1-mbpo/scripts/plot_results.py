"""
plot_results.py — MBPO 학습 결과 시각화
Usage:
  python scripts/plot_results.py --results-dir ./results
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str):
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.pkl') and 'MultiSeed' in fname:
            path = os.path.join(results_dir, fname)
            with open(path, 'rb') as f:
                return pickle.load(f), fname
    return None, None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default='./results')
    p.add_argument('--figures-dir', type=str, default='./figures')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    data, fname = load_results(args.results_dir)
    if data is None:
        print(f'No MultiSeed pkl found in {args.results_dir}')
        return

    all_rewards = data['all_rewards']
    all_steps   = data.get('all_steps', [None] * len(all_rewards))

    max_ep = max(len(r) for r in all_rewards)
    matrix = np.full((len(all_rewards), max_ep), np.nan)
    for i, r in enumerate(all_rewards):
        matrix[i, :len(r)] = r

    mean_r = np.nanmean(matrix, axis=0)
    std_r  = np.nanstd(matrix, axis=0)
    xs     = np.arange(1, max_ep + 1)

    # Episode reward
    fig, ax = plt.subplots(1, 1)
    ax.plot(xs, mean_r, label='mean reward')
    ax.fill_between(xs, mean_r - std_r, mean_r + std_r, alpha=0.3, label='±1 std')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('MBPO — Hopper-v5 (mean ± std)')
    ax.legend()
    out = os.path.join(args.figures_dir, 'episode_reward_mean_std.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    # eval_reward_avg per seed (if available)
    all_metrics = data.get('all_metrics', [])
    if all_metrics:
        eval_rewards = []
        for m_list in all_metrics:
            evals = [m.get('eval_reward_avg', float('nan')) for m in m_list]
            eval_rewards.append(evals)

        max_ep2 = max(len(e) for e in eval_rewards)
        em = np.full((len(eval_rewards), max_ep2), np.nan)
        for i, e in enumerate(eval_rewards):
            em[i, :len(e)] = e

        mean_e = np.nanmean(em, axis=0)
        std_e  = np.nanstd(em, axis=0)
        xs2    = np.arange(1, max_ep2 + 1)

        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(xs2, mean_e, label='eval reward (mean)')
        ax2.fill_between(xs2, mean_e - std_e, mean_e + std_e, alpha=0.3, label='±1 std')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Eval Reward')
        ax2.set_title('MBPO — Hopper-v5 Eval Reward (mean ± std)')
        ax2.legend()
        out2 = os.path.join(args.figures_dir, 'eval_reward_mean_std.png')
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved: {out2}')


if __name__ == '__main__':
    main()
