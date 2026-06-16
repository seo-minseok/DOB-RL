"""Plot per-seed and mean±std figures for Cycle_1-mbpo mbpo_uncert results."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results/mbpo_uncert")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../figures/mbpo_uncert")
SEEDS = list(range(1, 9))

METRICS = {
    "reward":                   "Episode Reward",
    "eval_reward_avg":          "Eval Reward (avg)",
    "ep_steps":                 "Episode Steps",
    "env_return":               "Env Return",
    "nominal_error_avg":        "Nominal Error (avg)",
    "residual_error_avg":       "Residual Error (avg)",
    "dhat_norm_avg":            "DOB d̂ Norm (avg)",
    "real_env_uncert_avg":      "Real Env Uncertainty (avg)",
    "res_net_loss":             "Residual Net Loss",
    "td_loss_avg":              "TD Loss (avg)",
    "buffer_uncert_avg":        "Buffer Uncertainty (avg)",
    "buffer_uncert_std":        "Buffer Uncertainty (std)",
    "buffer_uncert_p50":        "Buffer Uncertainty p50",
    "buffer_uncert_p75":        "Buffer Uncertainty p75",
    "buffer_uncert_p90":        "Buffer Uncertainty p90",
    "buffer_uncert_p95":        "Buffer Uncertainty p95",
    "resnet_sample_uncert_avg": "ResNet Sample Uncertainty (avg)",
    "rollout_avg_horizon":      "Rollout Avg Horizon",
}


def save_figure(fig, path):
    axes = fig.get_axes()
    assert len(axes) == 1, f"Only single-Axes figures allowed, got {len(axes)}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def smooth(values, window=10):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot_per_seed(df, metric, label, out_path):
    x = df["total_steps"].values
    y = df[metric].values
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(x[valid], y[valid], color="steelblue", alpha=0.4, linewidth=0.8, label="raw")
    ax.plot(x[valid], smooth(y[valid]), color="steelblue", linewidth=1.8, label="smoothed")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_path)


def plot_mean_std(dfs, metric, label, out_path):
    all_steps = np.concatenate([df["total_steps"].values for df in dfs])
    x_min, x_max = int(all_steps.min()), int(all_steps.max())
    x_common = np.linspace(x_min, x_max, 500)

    interp_values = []
    for df in dfs:
        x = df["total_steps"].values
        y = df[metric].values
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            continue
        y_interp = np.interp(x_common, x[valid], y[valid])
        interp_values.append(y_interp)

    if len(interp_values) < 2:
        return

    arr = np.array(interp_values)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(x_common, mean, color="steelblue", linewidth=2.0, label=f"mean (n={len(interp_values)})")
    ax.fill_between(x_common, mean - std, mean + std, color="steelblue", alpha=0.2, label="±1 std")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel(label)
    ax.set_title(f"{label} — 8-seed mean±std")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_path)


def main():
    dfs = {}
    for seed in SEEDS:
        csv_path = os.path.join(RESULTS_DIR, f"seed_{seed}_progress.csv")
        if os.path.exists(csv_path):
            dfs[seed] = pd.read_csv(csv_path)

    print(f"Loaded {len(dfs)} seeds: {sorted(dfs.keys())}")

    # Per-seed figures → figures/mbpo_uncert/seed_N/{metric}.png
    for seed, df in dfs.items():
        seed_dir = os.path.join(FIGURES_DIR, f"seed_{seed}")
        for metric, label in METRICS.items():
            if metric not in df.columns:
                continue
            out_path = os.path.join(seed_dir, f"{metric}.png")
            plot_per_seed(df, metric, label, out_path)
        print(f"  seed_{seed}: done")

    # Mean±std figures → figures/mbpo_uncert/mean_std/{metric}.png
    mean_dir = os.path.join(FIGURES_DIR, "mean_std")
    all_dfs = list(dfs.values())
    for metric, label in METRICS.items():
        if not any(metric in df.columns for df in all_dfs):
            continue
        out_path = os.path.join(mean_dir, f"{metric}.png")
        plot_mean_std(all_dfs, metric, label, out_path)
    print("  mean_std: done")

    print("All figures saved.")


if __name__ == "__main__":
    main()
