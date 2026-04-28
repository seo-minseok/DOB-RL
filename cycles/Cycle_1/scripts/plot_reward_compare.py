"""
plot_reward_compare.py — baseline / ablation / dqn / mbrl 네 조건의
multi-seed reward를 단일 Axes에 mean ± std 오버레이로 시각화.

출력: figures/compare/reward_multiseed_compare.png
실행: python scripts/plot_reward_compare.py  (cycles/Cycle_1/ 기준)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
RESULTS_DIR  = Path("results")
OUT_PATH     = Path("figures/compare/reward_multiseed_compare.png")
SMOOTH_WIN   = 10   # rolling-mean window (에피소드 단위)

CONFIGS = [
    dict(
        key   = "baseline",
        file  = "DOB_MBRL_MultiSeed_Result.csv",
        label = "DOB-MBRL (Baseline)",
        color = "#2166AC",   # 파란색
    ),
    dict(
        key   = "ablation",
        file  = "DOB_MBRL_MultiSeed_Result_Ablation.csv",
        label = "DOB-MBRL (Ablation, Uniform Sampling)",
        color = "#D6604D",   # 주황-빨강
    ),
    dict(
        key   = "dqn",
        file  = "DQN_MultiSeed_Result.csv",
        label = "DQN",
        color = "#1A9641",   # 초록
    ),
    dict(
        key   = "mbrl",
        file  = "MBRL_MultiSeed_Result.csv",
        label = "MBRL (Nominal)",
        color = "#984EA3",   # 보라
    ),
]


def smooth(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(window=win, min_periods=1, center=True).mean()


def load_stats(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """에피소드별 mean / std 반환 (seed 축 집계)."""
    df = pd.read_csv(csv_path)
    grouped = df.groupby("episode")["reward"]
    mean = grouped.mean()
    std  = grouped.std().fillna(0)
    episodes = mean.index.to_numpy()
    return episodes, mean.to_numpy(), std.to_numpy()


# ---------------------------------------------------------------------------
# 플롯
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for cfg in CONFIGS:
    episodes, mean, std = load_stats(RESULTS_DIR / cfg["file"])
    mean_s = smooth(pd.Series(mean), SMOOTH_WIN).to_numpy()
    std_s  = smooth(pd.Series(std),  SMOOTH_WIN).to_numpy()

    ax.plot(episodes, mean_s, color=cfg["color"], linewidth=2.0, label=cfg["label"])
    ax.fill_between(
        episodes,
        mean_s - std_s,
        mean_s + std_s,
        color=cfg["color"],
        alpha=0.15,
    )

ax.set_xlabel("Episode", fontsize=13)
ax.set_ylabel("Reward", fontsize=13)
ax.set_title("Multi-Seed Reward Comparison (mean ± std, smoothed)", fontsize=14)
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlim(1, episodes[-1])

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
plt.close(fig)
print(f"저장 완료: {OUT_PATH}")
