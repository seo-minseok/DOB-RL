"""
Multi_Seed_DOB_Exp.py

PyTorch + Gymnasium implementation of Multi_Seed_DOB_Exp.m
Runs train_DOB_core across multiple random seeds in parallel,
then visualises the mean ± std reward curve.
"""

import time
import pickle
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt

if os.name == 'nt' and os.environ.get('MUJOCO_GL', '').lower() == 'egl':
    os.environ.pop('MUJOCO_GL', None)

from train_DOB_core import train_DOB_core


# ============================================================
# Worker function (needed for multiprocessing on all platforms)
# ============================================================
def _run_single(args):
    run_idx, num_episodes = args
    print(f'[Run {run_idx:02d}] DOB-MBRL training started...')
    rewards, steps = train_DOB_core(run_idx, num_episodes)
    print(f'[Run {run_idx:02d}] Done.')
    return rewards, steps


# ============================================================
# Main experiment
# ============================================================
if __name__ == '__main__':
    # --------------------------------------------------
    # Experiment settings  (MATLAB defaults)
    # --------------------------------------------------
    num_runs     = int(os.getenv('DOB_NUM_RUNS', '16'))         # MATLAB: numRuns = 16
    max_episodes = int(os.getenv('DOB_MAX_EPISODES', '200'))    # MATLAB: numEpisodes = 200
    target_score = float(os.getenv('DOB_TARGET_SCORE', '480'))  # MATLAB: targetScore = 480

    print('==========================================================')
    print(f'DOB-MBRL Multi-Seed Training Started (Runs: {num_runs}, Ep: {max_episodes})')
    print('==========================================================\n')
    start_time = time.time()

    # --------------------------------------------------
    # Parallel training  (MATLAB: parfor)
    # --------------------------------------------------
    num_workers = min(num_runs, multiprocessing.cpu_count())
    args_list   = [(i + 1, max_episodes) for i in range(num_runs)]

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_run_single, args_list)
    except (OSError, PermissionError) as exc:
        print(f'Parallel execution unavailable ({exc}). Falling back to sequential execution.')
        results = [_run_single(args) for args in args_list]

    all_rewards = [r[0] for r in results]   # list of lists
    all_steps   = [r[1] for r in results]

    total_time = time.time() - start_time
    print(f'\n=== All training complete (Time: {total_time / 60:.1f} min) ===')

    # --------------------------------------------------
    # Save results  (MATLAB: save 'DOB_MBRL_MultiSeed_Result.mat')
    # --------------------------------------------------
    with open('DOB_MBRL_MultiSeed_Result.pkl', 'wb') as f:
        pickle.dump({'all_rewards': all_rewards, 'all_steps': all_steps}, f)
    print('Saved: DOB_MBRL_MultiSeed_Result.pkl')

    # --------------------------------------------------
    # Data pre-processing  (MATLAB: interpolation onto common step axis)
    # --------------------------------------------------
    max_total_step = max(
        max(steps) for steps in all_steps if steps
    )
    common_steps   = np.linspace(0, max_total_step, 1000)
    interp_rewards = np.full((num_runs, 1000), np.nan, dtype=np.float64)

    for i in range(num_runs):
        x_run = np.array([0] + list(all_steps[i]),   dtype=np.float64)
        y_run = np.array([0] + list(all_rewards[i]), dtype=np.float64)

        # MATLAB: unique(x_run)
        _, unique_idx = np.unique(x_run, return_index=True)
        x_run = x_run[unique_idx]
        y_run = y_run[unique_idx]

        if len(x_run) < 2:
            continue

        # MATLAB: interp1(x,y,commonSteps,'linear',NaN)
        vals = np.interp(common_steps, x_run, y_run, left=np.nan, right=np.nan)

        # MATLAB: vals(commonSteps > lastStep) = lastVal
        last_step = x_run[-1]
        last_val  = y_run[-1]
        vals[common_steps > last_step] = last_val

        interp_rewards[i, :] = vals

    mean_curve  = np.nanmean(interp_rewards, axis=0)
    std_curve   = np.nanstd( interp_rewards, axis=0)
    upper_curve = mean_curve + std_curve
    lower_curve = mean_curve - std_curve

    # --------------------------------------------------
    # Visualisation  (MATLAB: figure / fill / plot / yline)
    # MATLAB: 제안 기법은 검은색 계열로 표시
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.grid(True)

    # MATLAB: fill(x_fill, y_fill, 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none')
    ax.fill_between(common_steps, lower_curve, upper_curve,
                    color='black', alpha=0.2, linewidth=0, label='Std Dev')

    # MATLAB: plot(commonSteps, meanCurve, 'k-', 'LineWidth', 2.5)
    ax.plot(common_steps, mean_curve, 'k-', linewidth=2.5, label='Mean Reward')

    # MATLAB: yline(targetScore, 'k--', 'LineWidth', 1.5, 'Label', 'Target')
    ax.axhline(y=target_score, color='black', linestyle='--',
               linewidth=1.5, label='Target')

    ax.set_xlabel('Total Environmental Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Reward',            fontsize=12, fontweight='bold')
    ax.set_title(f'DOB-MBRL Performance (Avg over {num_runs} Seeds)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(-50, 550)

    plt.tight_layout()
    plt.savefig('DOB_MBRL_MultiSeed_Result.png', dpi=150)
    plt.show()
    print('Plot saved → DOB_MBRL_MultiSeed_Result.png')