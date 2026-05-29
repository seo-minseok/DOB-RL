"""
DOB-MBRL vs DQN vs MBRL(Nominal) 성능 비교 분석 스크립트
분석 대상: cycles/Cycle_1/results/ 내 CSV 파일 (16 seeds × 200 episodes)
Ablation(Uniform Sampling)은 분석 제외
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── 데이터 로드 ───────────────────────────────────────────────────────────────
BASE = "cycles/Cycle_1/results"

dob  = pd.read_csv(f"{BASE}/DOB_MBRL_MultiSeed_Result.csv")
dqn  = pd.read_csv(f"{BASE}/DQN_MultiSeed_Result.csv")
mbrl = pd.read_csv(f"{BASE}/MBRL_MultiSeed_Result.csv")

# 컬럼 통일
for df in [dob, dqn, mbrl]:
    df.rename(columns={"seed": "seed", "episode": "episode", "reward": "reward"}, inplace=True)

SEEDS = sorted(dob["seed"].unique())   # [1..16]
N_SEEDS = len(SEEDS)
TARGET_REWARD = 400.0

def get_reward_matrix(df):
    """shape: (n_seeds, n_episodes)  — 에피소드 1~200"""
    seeds = sorted(df["seed"].unique())
    episodes = sorted(df["episode"].unique())
    mat = np.full((len(seeds), len(episodes)), np.nan)
    for i, s in enumerate(seeds):
        sub = df[df["seed"] == s].sort_values("episode")
        for j, ep in enumerate(episodes):
            row = sub[sub["episode"] == ep]["reward"].values
            if len(row) > 0:
                mat[i, j] = row[0]
    return mat, seeds, episodes

dob_mat,  _, eps = get_reward_matrix(dob)
dqn_mat,  _, _   = get_reward_matrix(dqn)
mbrl_mat, _, _   = get_reward_matrix(mbrl)

eps = np.array(eps)   # 1..200

# ─── 헬퍼 ─────────────────────────────────────────────────────────────────────
def slice_idx(eps, lo, hi):
    return np.where((eps >= lo) & (eps <= hi))[0]

def first_reach(mat, target=TARGET_REWARD):
    """시드별 목표 보상 최초 도달 에피소드 (0-index ep 기준 → +1 for 1-based)"""
    results = []
    for row in mat:
        found = np.where(row >= target)[0]
        results.append(int(found[0] + 1) if len(found) > 0 else None)
    return results

def segment_stats(mat, idx):
    """구간 내 시드 평균 reward의 평균/표준편차/min/max/CV"""
    seg = mat[:, idx]                    # (n_seeds, n_ep_in_segment)
    seed_means = np.nanmean(seg, axis=1) # (n_seeds,)
    return {
        "mean":  float(np.nanmean(seed_means)),
        "std":   float(np.nanstd(seed_means, ddof=1)),
        "min":   float(np.nanmin(seed_means)),
        "max":   float(np.nanmax(seed_means)),
        "cv":    float(np.nanstd(seed_means, ddof=1) / np.nanmean(seed_means))
               if np.nanmean(seed_means) != 0 else float("nan"),
        "seed_means": seed_means,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 분석 1: 수렴 속도 비교
# ═══════════════════════════════════════════════════════════════════════════════

dob_reach  = first_reach(dob_mat)
dqn_reach  = first_reach(dqn_mat)
mbrl_reach = first_reach(mbrl_mat)

def reach_stats(reach_list):
    valid = [x for x in reach_list if x is not None]
    failed = len(reach_list) - len(valid)
    return {
        "mean":   float(np.mean(valid)) if valid else float("nan"),
        "std":    float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan"),
        "failed": failed,
        "valid":  valid,
    }

dob_rs  = reach_stats(dob_reach)
dqn_rs  = reach_stats(dqn_reach)
mbrl_rs = reach_stats(mbrl_reach)

# 에피소드 50~100 구간 구간 통계
idx_50_100 = slice_idx(eps, 50, 100)
dob_50_100  = segment_stats(dob_mat,  idx_50_100)
dqn_50_100  = segment_stats(dqn_mat,  idx_50_100)
mbrl_50_100 = segment_stats(mbrl_mat, idx_50_100)

# 통계 검정: paired Wilcoxon signed-rank test (16쌍)
def pairwise_wilcoxon(a, b):
    """시드별 구간 평균으로 쌍 검정"""
    try:
        stat, p = stats.wilcoxon(a, b)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")

def pairwise_ttest(a, b):
    stat, p = stats.ttest_rel(a, b)
    return float(stat), float(p)

sm_dob  = dob_50_100["seed_means"]
sm_dqn  = dqn_50_100["seed_means"]
sm_mbrl = mbrl_50_100["seed_means"]

wilc_dob_dqn   = pairwise_wilcoxon(sm_dob, sm_dqn)
wilc_dob_mbrl  = pairwise_wilcoxon(sm_dob, sm_mbrl)
wilc_dqn_mbrl  = pairwise_wilcoxon(sm_dqn, sm_mbrl)
t_dob_dqn      = pairwise_ttest(sm_dob, sm_dqn)
t_dob_mbrl     = pairwise_ttest(sm_dob, sm_mbrl)
t_dqn_mbrl     = pairwise_ttest(sm_dqn, sm_mbrl)

# ═══════════════════════════════════════════════════════════════════════════════
# 분석 2: 후반부 성능 붕괴 분석
# ═══════════════════════════════════════════════════════════════════════════════

idx_100_120 = slice_idx(eps, 100, 120)
idx_170_200 = slice_idx(eps, 170, 200)

def collapse_analysis(mat, label):
    peak_means  = np.nanmean(mat[:, idx_100_120], axis=1)  # (n_seeds,)
    late_means  = np.nanmean(mat[:, idx_170_200], axis=1)
    drop        = peak_means - late_means                   # 양수 = 하락
    drop_pct    = drop / np.where(peak_means != 0, peak_means, np.nan) * 100
    collapsed   = int(np.sum(drop_pct >= 30))               # 피크 대비 30% 이상 하락

    results = []
    for i, seed in enumerate(SEEDS):
        results.append({
            "seed":       seed,
            "peak_mean":  float(peak_means[i]),
            "late_mean":  float(late_means[i]),
            "drop":       float(drop[i]),
            "drop_pct":   float(drop_pct[i]),
            "collapsed":  bool(drop_pct[i] >= 30),
        })
    return results, collapsed, float(np.nanmean(drop)), float(np.nanmean(drop_pct))

dob_collapse,  dob_ncol,  dob_avg_drop,  dob_avg_drop_pct  = collapse_analysis(dob_mat,  "DOB-MBRL")
dqn_collapse,  dqn_ncol,  dqn_avg_drop,  dqn_avg_drop_pct  = collapse_analysis(dqn_mat,  "DQN")
mbrl_collapse, mbrl_ncol, mbrl_avg_drop, mbrl_avg_drop_pct = collapse_analysis(mbrl_mat, "MBRL")

# 붕괴 시드 공통 패턴: 급격한 하락 에피소드 탐색
def find_drop_episode(mat, seed_idx, window=5, threshold=50):
    """연속 window 에피소드 내 reward 하락 폭이 threshold 이상인 최초 에피소드 탐색"""
    row = mat[seed_idx]
    for i in range(window, len(row)):
        prev_mean = np.nanmean(row[i-window:i])
        curr      = row[i]
        if not np.isnan(curr) and prev_mean - curr > threshold:
            return int(eps[i])
    return None

collapsed_seed_info = []
for rec in dob_collapse:
    if rec["collapsed"]:
        sidx = SEEDS.index(rec["seed"])
        drop_ep = find_drop_episode(dob_mat, sidx)
        collapsed_seed_info.append({
            "seed":    rec["seed"],
            "drop_ep": drop_ep,
            "drop_pct": rec["drop_pct"],
        })

# ═══════════════════════════════════════════════════════════════════════════════
# 분석 3: 구간별 통계 비교
# ═══════════════════════════════════════════════════════════════════════════════

SEGMENTS = [
    ("초기 학습",  1,   50),
    ("빠른 성장", 51,  100),
    ("안정화",   101,  150),
    ("후반부",   151,  200),
]

seg_results = {}
for name, lo, hi in SEGMENTS:
    idx = slice_idx(eps, lo, hi)
    seg_results[name] = {
        "DOB-MBRL": segment_stats(dob_mat,  idx),
        "DQN":      segment_stats(dqn_mat,  idx),
        "MBRL":     segment_stats(mbrl_mat, idx),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 마크다운 보고서 생성
# ═══════════════════════════════════════════════════════════════════════════════

lines = []
A = lines.append

A("# DOB-MBRL 성능 비교 분석 보고서")
A("")
A("> **분석 기준일**: 2026-04-17  ")
A("> **분석 대상**: DOB-MBRL (Baseline), DQN, MBRL (Nominal)  ")
A("> **데이터**: 16 seeds × 200 episodes each  ")
A("> **Ablation(Uniform Sampling)**: 분석 제외  ")
A("")
A("---")
A("")

# ──────────────────────────────────────────────────────────────────────────────
A("## 분석 1: 수렴 속도 비교 (에피소드 50~100 구간)")
A("")
A("### 1-1. 목표 보상(400) 최초 도달 에피소드")
A("")
A("| 방법 | 평균 도달 에피소드 | 표준편차 | 미도달 시드 수 | 도달 시드 수 |")
A("|------|:-----------------:|:--------:|:--------------:|:------------:|")

rows = [
    ("DOB-MBRL",  dob_rs),
    ("DQN",       dqn_rs),
    ("MBRL",      mbrl_rs),
]
for label, rs in rows:
    mean_s = f"{rs['mean']:.2f}" if not np.isnan(rs['mean']) else "N/A"
    std_s  = f"{rs['std']:.2f}"  if not np.isnan(rs['std'])  else "N/A"
    reached = N_SEEDS - rs['failed']
    A(f"| {label} | {mean_s} | {std_s} | {rs['failed']} | {reached} |")

A("")

# 시드별 최초 도달 에피소드 상세
A("### 1-2. 시드별 목표 보상 최초 도달 에피소드")
A("")
header = "| Seed |" + " DOB-MBRL |" + " DQN |" + " MBRL |"
sep    = "|------|" + "----------:|" + "-----:|" + "------:|"
A(header)
A(sep)
for i, s in enumerate(SEEDS):
    d = str(dob_reach[i])   if dob_reach[i]  is not None else "미도달"
    q = str(dqn_reach[i])   if dqn_reach[i]  is not None else "미도달"
    m = str(mbrl_reach[i])  if mbrl_reach[i] is not None else "미도달"
    A(f"| {s} | {d} | {q} | {m} |")

A("")

A("### 1-3. 에피소드 50~100 구간 시드 평균 Reward")
A("")
A("| 방법 | 평균 | 표준편차 |")
A("|------|-----:|---------:|")
for label, st in [("DOB-MBRL", dob_50_100), ("DQN", dqn_50_100), ("MBRL", mbrl_50_100)]:
    A(f"| {label} | {st['mean']:.2f} | {st['std']:.2f} |")
A("")

A("### 1-4. 통계적 유의성 검정 (에피소드 50~100 구간)")
A("")
A("#### Wilcoxon Signed-Rank Test (paired, n=16)")
A("")
A("| 비교 쌍 | W 통계량 | p-value | 유의성(α=0.05) |")
A("|---------|--------:|--------:|:--------------:|")
def sig(p): return "**유의**" if p < 0.05 else "비유의"
A(f"| DOB-MBRL vs DQN  | {wilc_dob_dqn[0]:.2f}  | {wilc_dob_dqn[1]:.4f}  | {sig(wilc_dob_dqn[1])}  |")
A(f"| DOB-MBRL vs MBRL | {wilc_dob_mbrl[0]:.2f} | {wilc_dob_mbrl[1]:.4f} | {sig(wilc_dob_mbrl[1])} |")
A(f"| DQN vs MBRL      | {wilc_dqn_mbrl[0]:.2f}  | {wilc_dqn_mbrl[1]:.4f}  | {sig(wilc_dqn_mbrl[1])}  |")
A("")
A("#### Paired t-test (n=16)")
A("")
A("| 비교 쌍 | t 통계량 | p-value | 유의성(α=0.05) |")
A("|---------|--------:|--------:|:--------------:|")
A(f"| DOB-MBRL vs DQN  | {t_dob_dqn[0]:.2f}  | {t_dob_dqn[1]:.4f}  | {sig(t_dob_dqn[1])}  |")
A(f"| DOB-MBRL vs MBRL | {t_dob_mbrl[0]:.2f} | {t_dob_mbrl[1]:.4f} | {sig(t_dob_mbrl[1])} |")
A(f"| DQN vs MBRL      | {t_dqn_mbrl[0]:.2f}  | {t_dqn_mbrl[1]:.4f}  | {sig(t_dqn_mbrl[1])}  |")
A("")
A("#### 해석")
A("")
A("> 에피소드 50~100 구간에서 세 방법 간 성능 차이의 통계적 유의성을 확인한다. "
  f"DOB-MBRL의 평균 reward는 {dob_50_100['mean']:.2f}(±{dob_50_100['std']:.2f}), "
  f"DQN은 {dqn_50_100['mean']:.2f}(±{dqn_50_100['std']:.2f}), "
  f"MBRL은 {mbrl_50_100['mean']:.2f}(±{mbrl_50_100['std']:.2f})이다. "
  "목표 보상 도달 에피소드와 구간 평균을 종합하면, 수렴 속도 측면에서 세 방법 간 우열을 판단할 수 있다.")
A("")
A("---")
A("")

# ──────────────────────────────────────────────────────────────────────────────
A("## 분석 2: 후반부 성능 붕괴 분석 (에피소드 150~200 구간)")
A("")
A("### 2-1. 에피소드 100~120 → 170~200 구간 Reward 변화")
A("")
A("| 방법 | 100~120 평균 | 170~200 평균 | 평균 하락폭 | 평균 하락률(%) | 붕괴 시드 수(≥30% 하락) |")
A("|------|------------:|------------:|----------:|------------:|:--------------------:|")

def fmt(v): return f"{v:.2f}" if not np.isnan(v) else "N/A"
A(f"| DOB-MBRL | {fmt(np.nanmean(dob_mat[:,idx_100_120]))} | {fmt(np.nanmean(dob_mat[:,idx_170_200]))} | {fmt(dob_avg_drop)} | {fmt(dob_avg_drop_pct)} | {dob_ncol} |")
A(f"| DQN      | {fmt(np.nanmean(dqn_mat[:,idx_100_120]))} | {fmt(np.nanmean(dqn_mat[:,idx_170_200]))} | {fmt(dqn_avg_drop)} | {fmt(dqn_avg_drop_pct)} | {dqn_ncol} |")
A(f"| MBRL     | {fmt(np.nanmean(mbrl_mat[:,idx_100_120]))} | {fmt(np.nanmean(mbrl_mat[:,idx_170_200]))} | {fmt(mbrl_avg_drop)} | {fmt(mbrl_avg_drop_pct)} | {mbrl_ncol} |")
A("")

A("### 2-2. DOB-MBRL 시드별 붕괴 분석")
A("")
A("| Seed | 100~120 평균 | 170~200 평균 | 하락폭 | 하락률(%) | 붕괴 여부 |")
A("|------|------------:|------------:|------:|----------:|:--------:|")
for rec in dob_collapse:
    flag = "**붕괴**" if rec["collapsed"] else "-"
    A(f"| {rec['seed']} | {rec['peak_mean']:.2f} | {rec['late_mean']:.2f} | {rec['drop']:.2f} | {rec['drop_pct']:.2f} | {flag} |")
A("")

if collapsed_seed_info:
    A("### 2-3. DOB-MBRL 붕괴 시드 패턴 분석")
    A("")
    A("| Seed | 급격한 하락 탐지 에피소드 | 하락률(%) |")
    A("|------|:------------------------:|----------:|")
    for info in collapsed_seed_info:
        ep_str = str(info["drop_ep"]) if info["drop_ep"] is not None else "점진적 하락"
        A(f"| {info['seed']} | {ep_str} | {info['drop_pct']:.2f} |")
    A("")
    A("> *급격한 하락: 직전 5 에피소드 평균 대비 50 이상 reward 감소*")
    A("")

A("### 2-4. DQN, MBRL 같은 구간 성능 변화 비교")
A("")
A("| 방법 | 100~120 평균 | 170~200 평균 | 하락폭 | 하락률(%) |")
A("|------|------------:|------------:|------:|----------:|")
for label, mat in [("DOB-MBRL", dob_mat), ("DQN", dqn_mat), ("MBRL", mbrl_mat)]:
    p = float(np.nanmean(mat[:, idx_100_120]))
    l = float(np.nanmean(mat[:, idx_170_200]))
    d = p - l
    dp = d / p * 100 if p != 0 else float("nan")
    A(f"| {label} | {p:.2f} | {l:.2f} | {d:.2f} | {dp:.2f} |")
A("")

A("#### 해석")
A("")
A(f"> DOB-MBRL에서 붕괴(피크 대비 ≥30% 하락)로 분류된 시드는 {dob_ncol}개이며, "
  f"전체 평균 하락률은 {dob_avg_drop_pct:.2f}%이다. "
  f"반면 DQN은 {dqn_ncol}개 시드 붕괴(평균 {dqn_avg_drop_pct:.2f}% 하락), "
  f"MBRL은 {mbrl_ncol}개 시드 붕괴(평균 {mbrl_avg_drop_pct:.2f}% 하락)로 "
  "후반부 안정성에서 세 방법 간 차이를 보인다.")
A("")
A("---")
A("")

# ──────────────────────────────────────────────────────────────────────────────
A("## 분석 3: 구간별 평균/표준편차 비교")
A("")
for seg_name, lo, hi in SEGMENTS:
    A(f"### {seg_name} (에피소드 {lo}~{hi})")
    A("")
    A("| 방법 | 평균 | 표준편차 | 최소 시드 평균 | 최대 시드 평균 | 변동계수(CV) |")
    A("|------|-----:|---------:|-------------:|-------------:|------------:|")
    for method in ["DOB-MBRL", "DQN", "MBRL"]:
        st = seg_results[seg_name][method]
        cv_s = f"{st['cv']:.4f}" if not np.isnan(st['cv']) else "N/A"
        A(f"| {method} | {st['mean']:.2f} | {st['std']:.2f} | {st['min']:.2f} | {st['max']:.2f} | {cv_s} |")
    A("")

A("### 구간별 요약 (세 방법 동시 비교)")
A("")
A("| 구간 | 방법 | 평균 | CV |")
A("|------|------|-----:|---:|")
for seg_name, lo, hi in SEGMENTS:
    for method in ["DOB-MBRL", "DQN", "MBRL"]:
        st = seg_results[seg_name][method]
        cv_s = f"{st['cv']:.4f}" if not np.isnan(st['cv']) else "N/A"
        A(f"| {seg_name}({lo}~{hi}) | {method} | {st['mean']:.2f} | {cv_s} |")
A("")

A("#### 해석")
A("")
# 후반부 CV 비교
late_cvs = {m: seg_results["후반부"][m]["cv"] for m in ["DOB-MBRL", "DQN", "MBRL"]}
best_cv  = min(late_cvs, key=lambda x: late_cvs[x] if not np.isnan(late_cvs[x]) else 9999)
A(f"> 구간별 변동계수(CV)를 통해 시드 간 일관성을 평가할 수 있다. "
  f"후반부(151~200) 구간에서 CV가 가장 낮은 방법은 {best_cv}({late_cvs[best_cv]:.4f})로, "
  "시드 간 편차가 상대적으로 작아 안정적임을 나타낸다. "
  "CV가 높을수록 시드에 따라 성능 편차가 크며, 알고리즘의 재현성이 낮음을 의미한다.")
A("")
A("---")
A("")

# ──────────────────────────────────────────────────────────────────────────────
A("## 데이터 출처")
A("")
A("| 파일 | 행 수 | 시드 수 | 에피소드 수 |")
A("|------|------:|-------:|----------:|")
A(f"| DOB_MBRL_MultiSeed_Result.csv  | {len(dob)}  | {N_SEEDS} | {len(eps)} |")
A(f"| DQN_MultiSeed_Result.csv       | {len(dqn)}  | {N_SEEDS} | {len(eps)} |")
A(f"| MBRL_MultiSeed_Result.csv      | {len(mbrl)} | {N_SEEDS} | {len(eps)} |")

report = "\n".join(lines)

with open("cycles/Cycle_1/analysis_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("분석 완료 → cycles/Cycle_1/analysis_report.md 저장됨")

# ── 검증용 콘솔 출력 ────────────────────────────────────────────────────────
print("\n=== 분석 1: 수렴 속도 ===")
for label, rs in [("DOB-MBRL", dob_rs), ("DQN", dqn_rs), ("MBRL", mbrl_rs)]:
    print(f"  {label}: 평균 {rs['mean']:.2f} ep (±{rs['std']:.2f}), 미도달 {rs['failed']}개")

print("\n=== 분석 1: 50~100 구간 ===")
for label, st in [("DOB-MBRL", dob_50_100), ("DQN", dqn_50_100), ("MBRL", mbrl_50_100)]:
    print(f"  {label}: 평균 {st['mean']:.2f} ±{st['std']:.2f}")

print(f"\n=== 분석 1: Wilcoxon (DOB vs DQN)  W={wilc_dob_dqn[0]:.2f}, p={wilc_dob_dqn[1]:.4f}")
print(f"=== 분석 1: Wilcoxon (DOB vs MBRL) W={wilc_dob_mbrl[0]:.2f}, p={wilc_dob_mbrl[1]:.4f}")
print(f"=== 분석 1: Wilcoxon (DQN vs MBRL) W={wilc_dqn_mbrl[0]:.2f}, p={wilc_dqn_mbrl[1]:.4f}")

print(f"\n=== 분석 2: DOB-MBRL 붕괴 시드 수: {dob_ncol} ===")
print(f"=== 분석 2: DQN 붕괴 시드 수:      {dqn_ncol} ===")
print(f"=== 분석 2: MBRL 붕괴 시드 수:     {mbrl_ncol} ===")
