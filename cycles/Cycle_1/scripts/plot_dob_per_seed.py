"""
plot_dob_per_seed.py — DOB_MBRL_MultiSeed_Result.csv의 각 시드별 reward를
단일 Axes에 개별 곡선으로 시각화.

출력: figures/baseline/reward_per_seed.png
실행: python scripts/plot_dob_per_seed.py  (cycles/Cycle_1/ 기준)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
CSV_PATH = Path("results/DOB_MBRL_MultiSeed_Result.csv")
OUT_PATH = Path("figures/baseline/reward_per_seed.png")
SMOOTH_WIN = 10  # rolling-mean window (에피소드 단위)


def smooth(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(window=win, min_periods=1, center=True).mean()


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
seeds = sorted(df["seed"].unique())

# 색상: 시드 수에 맞게 컬러맵 생성
colors = cm.tab20(np.linspace(0, 1, len(seeds)))

# ---------------------------------------------------------------------------
# 플롯
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

for seed, color in zip(seeds, colors):
    seed_df = df[df["seed"] == seed].sort_values("episode")
    episodes = seed_df["episode"].to_numpy()
    reward_s = smooth(seed_df["reward"].reset_index(drop=True), SMOOTH_WIN).to_numpy()
    ax.plot(episodes, reward_s, color=color, linewidth=1.2, alpha=0.8, label=f"Seed {seed}")

ax.set_xlabel("Episode", fontsize=13)
ax.set_ylabel("Reward (smoothed)", fontsize=13)
ax.set_title(f"DOB-MBRL Per-Seed Reward (smoothed, window={SMOOTH_WIN})", fontsize=14)
ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.7)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlim(1, episodes[-1])

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
plt.close(fig)
print(f"저장 완료: {OUT_PATH}")
