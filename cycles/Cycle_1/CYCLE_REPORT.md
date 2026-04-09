# Cycle 1 실험 보고서

> 생성일: 2026-04-09
> 시작점: base

---

## 1. 변경점 요약

Base 코드를 그대로 이식한 검증 Cycle. 코드 구조 자체의 변경은 없으며, 이 Cycle에서 수행한 실질적 작업은 다음과 같다.

- **Ablation 기능 추가**: `train_residual_dx_model_dob`의 uncertainty-weighted sampling을 uniform sampling으로 대체하는 ablation 모드 구현 (`use_uncertainty_sampling` 플래그)
- **실험 인프라 정비**: checkpoint/results/figures 저장 경로를 `baseline/` · `ablation/` 하위로 분리, `run_multi_seed.py`의 results 경로 버그 수정
- **시각화 확장**: ablation 단독 figure 및 baseline vs ablation 비교 오버레이 figure 생성 기능 추가

---

## 2. 변경 상세

| 파일 | 변경 내용 |
|---|---|
| `dob_mbrl/training/config.py` | `use_uncertainty_sampling: bool = True` 필드 추가 |
| `dob_mbrl/training/model_learning.py` | `train_residual_dx_model_dob`에 `use_uncertainty_sampling` 인자 추가, uniform/weighted sampling 분기 구현 |
| `dob_mbrl/training/trainer.py` | `cfg.use_uncertainty_sampling`을 `train_residual_dx_model_dob` 호출 시 전달 |
| `run_multi_seed.py` | `--ablation` 플래그 추가, results 저장 경로를 `__file__` 기준으로 고정 (경로 버그 수정), 기본 checkpoint-dir을 `./checkpoints/baseline`으로 변경 |
| `main.py` | 기본 checkpoint-dir을 `./checkpoints/baseline`으로 변경 |
| `scripts/plot_results.py` | `--ablation` 플래그 (ablation 단독 figure), `--compare` 플래그 (baseline vs ablation 오버레이) 추가 |

---

## 3. 하이퍼파라미터

| 항목 | 값 |
|---|---|
| `num_episodes` | 200 |
| `max_steps_per_ep` | 500 |
| `warm_start_samples` | 200 |
| `lr_critic` | 1e-3 (Adam) |
| `discount_factor` | 0.99 |
| `tau` | 0.005 |
| `update_interval` | 10 |
| `num_gradient_steps` | 2 |
| `epsilon` / `epsilon_min` / `epsilon_decay` | 1.0 / 0.01 / 0.005 |
| `buffer_size` | 100,000 |
| `mini_batch_size` | 256 |
| `num_epochs` (residual) | 5 |
| `real_ratio` | 0.2 |
| `dob_w` | 0.1 |
| `num_rbf_centers` / `rbf_width` | 600 / 0.1 |
| `lr_rbf` / `lr_residual` | 0.5 / 0.01 |
| `max_horizon_length` | 10 |
| `uncertainty_threshold` | 0.1 |
| `num_generate_sample_iteration` | 20 |
| `epsilon_min_model` | 0.1 |
| `num_seeds` | 16 |

---

## 4. 학습 결과

### 4.1 Baseline (uncertainty-weighted sampling)

| 지표 | 값 |
|---|---|
| 시드 수 | 16 |
| 에피소드 수 | 200 |
| Final 10-ep avg — mean ± std | **280.59 ± 162.29** |
| Final 10-ep avg — max / min | 497.84 / 1.49 |
| Seeds ≥ 480 (최종 유지) | **2 / 16** |
| 전체 에피소드 reward — mean ± std | 327.00 ± 184.42 |
| 총 환경 스텝 — mean (min / max) | 69,672 (54,245 / 87,032) |
| 480 첫 도달 시드 수 | **16 / 16** |
| 480 첫 도달 에피소드 — mean / median | 74.8 / 71.0 |

시드별 Final 10-ep avg:

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Reward | 1.5 | 497.8 | 407.9 | 314.3 | 193.2 | 83.6 | 339.3 | 421.6 | 417.9 | 41.0 | 418.5 | 33.9 | 485.5 | 260.4 | 359.2 | 213.7 |

### 4.2 Ablation (uniform sampling)

| 지표 | 값 |
|---|---|
| 시드 수 | 16 |
| 에피소드 수 | 200 |
| Final 10-ep avg — mean ± std | **323.65 ± 172.29** |
| Final 10-ep avg — max / min | 496.06 / -2.62 |
| Seeds ≥ 480 (최종 유지) | **2 / 16** |
| 전체 에피소드 reward — mean ± std | 293.41 ± 195.63 |
| 총 환경 스텝 — mean (min / max) | 62,771 (30,328 / 75,834) |
| 480 첫 도달 시드 수 | **15 / 16** |
| 480 첫 도달 에피소드 — mean / median | 75.7 / 73.0 |

시드별 Final 10-ep avg:

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Reward | 172.4 | -2.6 | 456.2 | 496.1 | 431.2 | 81.9 | 468.5 | 429.5 | 391.4 | 467.4 | 61.3 | 487.8 | 371.6 | 299.4 | 97.5 | 468.8 |

### 4.3 Baseline vs Ablation 비교 요약

| 지표 | Baseline | Ablation | 해석 |
|---|---|---|---|
| Final 10-ep avg (mean) | 280.59 | **323.65** | Ablation이 수치상 높음 |
| Final 10-ep avg (std) | 162.29 | 172.29 | 두 조건 모두 분산 매우 큼 |
| Seeds ≥ 480 (최종) | 2 / 16 | 2 / 16 | 동일 |
| 전체 ep reward (mean) | **327.00** | 293.41 | Baseline이 학습 전체 평균 높음 |
| 총 환경 스텝 (mean) | **69,672** | 62,771 | Baseline이 에피소드당 생존 스텝 긺 |
| 480 첫 도달 | **16/16** | 15/16 | 거의 동일 |
| 수렴 에피소드 (mean) | **74.8** | 75.7 | 거의 동일 |

### 4.4 Figure 목록

| 파일 | 내용 |
|---|---|
| `figures/baseline/multiseed_mean_std.png` | Baseline 16-seed mean ± std (환경 스텝 기준) |
| `figures/ablation/multiseed_mean_std.png` | Ablation 16-seed mean ± std (환경 스텝 기준) |
| `figures/compare_baseline_vs_ablation.png` | Baseline vs Ablation 오버레이 비교 |

### 4.5 객관적 평가 및 의견

**uncertainty-weighted sampling의 효과는 이 실험 범위에서 뚜렷하지 않다.**

가장 주목할 수치는 "480 첫 도달 시드 수"다. Baseline 16/16, Ablation 15/16으로 두 조건 모두 학습 중 고성능에 도달하지만, 200 에피소드 종료 시점에 ≥ 480을 유지하는 시드는 양쪽 모두 2/16에 불과하다. 이는 **수렴 속도나 sampling 전략의 차이보다 학습 후반 안정성이 훨씬 큰 병목**임을 시사한다.

Final 10-ep avg의 mean만 보면 Ablation(323.65)이 Baseline(280.59)보다 높아 보이지만, std가 각각 172/162로 매우 크고, 시드별 편차도 극심하다 (Baseline: 1.5~497.8, Ablation: -2.6~496.1). 이 수준의 분산에서 두 조건의 평균 차이(약 43점)는 통계적으로 유의미하다고 보기 어렵다.

Ablation의 총 환경 스텝이 더 짧은 점(62,771 vs 69,672)은 흥미롭다. 에피소드 수는 동일(200)한데 총 스텝이 적다는 것은 Ablation 조건에서 에피소드당 생존 스텝이 짧았다는 의미이므로, 학습 후반부에서 Baseline보다 더 불안정했을 가능성이 있다.

**결론**: uncertainty-weighted sampling 자체의 유무보다, 학습이 고성능에 도달한 후 이를 유지하지 못하는 안정성 문제가 핵심이다. 다음 Cycle에서는 안정성 향상에 초점을 맞추는 것이 타당하다.

---

## 5. 관찰 및 교훈

<!-- ★ 사람이 직접 작성 -->

---

## 6. 메모리 업데이트 제안 (선택)

아래 내용을 `memory.md`의 "실험 교훈" 섹션에 추가를 제안한다 (승인 시 반영):

- **Cycle 1 교훈**: uncertainty-weighted sampling 유무보다 학습 후반 안정성이 더 큰 병목. 두 조건 모두 200 에피소드 내 16시드 중 2개만 ≥ 480 유지. 수렴 속도(~75 ep)는 동일. 다음 Cycle에서는 안정성(성능 유지) 향상 방안을 우선 탐색할 것.
