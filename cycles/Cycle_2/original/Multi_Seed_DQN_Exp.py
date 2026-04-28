"""
Multi_Seed_DQN_Exp.py

PyTorch + Gymnasium implementation of Multi_Seed_DQN_Exp.m
Runs train_DQN_core across multiple random seeds in parallel,
then visualises the mean +/- std reward curve.
"""

import multiprocessing
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train_DQN_core import train_DQN_core


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.getenv("DQN_RESULTS_DIR", str(PROJECT_ROOT / "Python" / "DQN")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _run_single(args):
    run_idx, num_episodes = args
    print(f"[Run {run_idx}] DQN training started...")
    rewards, steps = train_DQN_core(run_idx, num_episodes)
    print(f"[Run {run_idx}] Done.")
    return rewards, steps


if __name__ == "__main__":
    num_runs     = 16
    max_episodes = 200
    target_score = 480

    print(f"=== DQN Multi-Seed Training Started (Runs: {num_runs}) ===")
    start_time = time.time()

    num_workers = min(num_runs, multiprocessing.cpu_count())
    args_list   = [(i + 1, max_episodes) for i in range(num_runs)]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_run_single, args_list)

    all_rewards = [r[0] for r in results]
    all_steps   = [r[1] for r in results]

    total_time = time.time() - start_time
    print(f"=== Training Complete (Time: {total_time / 60:.1f} min) ===")

    result_pkl_path = RESULTS_DIR / "DQN_MultiSeed_Result.pkl"
    with open(result_pkl_path, "wb") as f:
        pickle.dump({"all_rewards": all_rewards, "all_steps": all_steps}, f)

    # =========================================================
    # Data preprocessing & visualisation
    # =========================================================

    # 1. Common X-axis (Total Steps)
    max_total_step = max(max(steps) for steps in all_steps if steps)
    common_steps   = np.linspace(0, max_total_step, 1000)
    interp_rewards = np.full((num_runs, 1000), np.nan, dtype=np.float64)

    # 2. Interpolation — no extrapolation; flat-fill after last step
    for i in range(num_runs):
        x_run = np.array([0] + list(all_steps[i]),   dtype=np.float64)
        y_run = np.array([0] + list(all_rewards[i]), dtype=np.float64)

        # Remove duplicates (MATLAB: unique(x_run))
        _, unique_idx = np.unique(x_run, return_index=True)
        x_run = x_run[unique_idx]
        y_run = y_run[unique_idx]

        if len(x_run) < 2:
            continue

        # (1) In-range interpolation (linear), NaN outside
        vals = np.interp(common_steps, x_run, y_run, left=np.nan, right=np.nan)

        # (2) Flat-fill beyond last step (MATLAB: vals(commonSteps > lastStep) = lastVal)
        last_step = x_run[-1]
        last_val  = y_run[-1]
        vals[common_steps > last_step] = last_val

        interp_rewards[i, :] = vals

    # 3. Statistics (omit NaN — MATLAB: 'omitnan')
    mean_curve  = np.nanmean(interp_rewards, axis=0)
    std_curve   = np.nanstd(interp_rewards,  axis=0)
    upper_curve = mean_curve + std_curve
    lower_curve = mean_curve - std_curve

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.grid(True)

    ax.fill_between(
        common_steps,
        lower_curve,
        upper_curve,
        color="red",
        alpha=0.2,
        linewidth=0,
        label="Std Dev",
    )
    ax.plot(common_steps, mean_curve, "r-", linewidth=2, label="Mean Reward")
    ax.axhline(y=target_score, color="black", linestyle="--", linewidth=1, label="Target")

    ax.set_xlabel("Total Environmental Steps")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"DQN Performance (Avg over {num_runs} Seeds)")
    ax.legend(loc="lower right")
    ax.set_ylim(-50, 550)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "DQN_MultiSeed_Result.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved -> {plot_path}")
