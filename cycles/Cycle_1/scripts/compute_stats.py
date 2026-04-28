import pandas as pd
import numpy as np

# Correct mapping
baseline_df = pd.read_csv('cycles/Cycle_1/results/DOB_MBRL_MultiSeed_Result.csv')
dqn_df      = pd.read_csv('cycles/Cycle_1/results/DQN_MultiSeed_Result.csv')
mbrl_df     = pd.read_csv('cycles/Cycle_1/results/MBRL_MultiSeed_Result.csv')

methods = {
    'Baseline': baseline_df,
    'DQN':      dqn_df,
    'MBRL':     mbrl_df,
}

# ─────────────────────────────────────────
# 1. Per-seed reward stats (ep 1–100)
# ─────────────────────────────────────────
print("=== Per-seed reward stats (ep 1-100) ===")
per_seed_stats = {}
for method, df in methods.items():
    per_seed_stats[method] = {}
    print(f"\n-- {method} --")
    for seed in sorted(df.seed.unique()):
        sdf = df[(df.seed == seed) & (df.episode >= 1) & (df.episode <= 100)]
        mean_r = sdf['reward'].mean()
        std_r  = sdf['reward'].std(ddof=1) if len(sdf) > 1 else 0.0
        per_seed_stats[method][seed] = {'mean_reward': mean_r, 'std_reward': std_r, 'n_ep': len(sdf)}
        print(f"  Seed{seed:2d}: mean={mean_r:8.3f}  std={std_r:8.3f}  n_ep={len(sdf)}")

# ─────────────────────────────────────────
# 2. TD loss (last 10%)
# ─────────────────────────────────────────
print("\n\n=== TD loss (last 10%) ===")
td_stats = {}
for method, df in methods.items():
    td_stats[method] = {}
    print(f"\n-- {method} --")
    if 'td_loss_avg' not in df.columns:
        print("  [no td_loss_avg column]")
        for seed in sorted(df.seed.unique()):
            td_stats[method][seed] = {'mean_td': float('nan'), 'max_td': float('nan')}
        continue
    for seed in sorted(df.seed.unique()):
        sdf = df[df.seed == seed].sort_values('episode')
        n = len(sdf)
        cutoff = int(n * 0.9)  # keep last 10%
        tail = sdf.iloc[cutoff:]
        td_vals = tail['td_loss_avg'].dropna()
        if len(td_vals) == 0:
            mean_td = max_td = float('nan')
        else:
            mean_td = td_vals.mean()
            max_td  = td_vals.max()
        td_stats[method][seed] = {'mean_td': mean_td, 'max_td': max_td}
        print(f"  Seed{seed:2d}: mean_td={mean_td if np.isnan(mean_td) else f'{mean_td:10.4f}'}  "
              f"max_td={max_td if np.isnan(max_td) else f'{max_td:10.4f}'}")

# ─────────────────────────────────────────
# 3. Cross-seed summary
# ─────────────────────────────────────────
print("\n\n=== Cross-seed summary ===")
cross = {}
for method in methods:
    means = [v['mean_reward'] for v in per_seed_stats[method].values()]
    stds  = [v['std_reward']  for v in per_seed_stats[method].values()]
    mom   = np.mean(means)
    som   = np.std(means, ddof=1)
    mos   = np.mean(stds)
    thresh = mom + 1.5 * som
    outliers = [s for s, v in per_seed_stats[method].items() if v['mean_reward'] > thresh]
    cross[method] = {
        'mean_of_means': mom,
        'std_of_means':  som,
        'mean_of_stds':  mos,
        'outliers':      outliers,
    }
    print(f"  {method}: mean_of_means={mom:.3f}  std_of_means={som:.3f}  "
          f"mean_of_stds={mos:.3f}  outliers={outliers}")

# ─────────────────────────────────────────
# 4. TD loss cross-seed
# ─────────────────────────────────────────
print("\n\n=== TD loss cross-seed ===")
td_cross = {}
for method in methods:
    vals     = [v['mean_td'] for v in td_stats[method].values() if not np.isnan(v['mean_td'])]
    max_vals = [v['max_td']  for v in td_stats[method].values() if not np.isnan(v['max_td'])]
    if vals:
        avg_td  = np.mean(vals)
        std_td  = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        max_all = max(max_vals)
    else:
        avg_td = std_td = max_all = float('nan')
    diverged = (max_all > 100.0) if not np.isnan(max_all) else False
    td_cross[method] = {'avg_td': avg_td, 'std_td': std_td, 'max_td': max_all, 'diverged': diverged}
    print(f"  {method}: avg_td={avg_td}  std_td={std_td}  max_td={max_all}  diverged={diverged}")
