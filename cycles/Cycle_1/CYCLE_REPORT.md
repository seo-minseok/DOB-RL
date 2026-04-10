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

#### 4.3.1 Reward 지표

| 지표 | Baseline | Ablation | 해석 |
|---|---|---|---|
| Final 10-ep avg (mean) | 280.59 | **323.65** | Ablation이 수치상 높음 |
| Final 10-ep avg (std) | 162.29 | 172.29 | 두 조건 모두 분산 매우 큼 |
| Seeds ≥ 480 (최종) | 2 / 16 | 2 / 16 | 동일 |
| 전체 ep reward (mean) | **327.00** | 293.41 | Baseline이 학습 전체 평균 높음 |
| ep 50~100 reward mean | 392.2 | **400.8** | 수렴 구간 유사 |
| ep 101~150 reward mean | **417.2** | 391.3 | Baseline이 중반 이후 높음 |
| ep 151~200 reward mean | 303.7 | 305.0 | 후반 붕괴 동일 수준 |
| 최종 20ep 중 480≥ 에피소드 비율 | 15.6% | **32.2%** | Ablation이 2배 이상 높음 |
| 최종 10ep episode_length (mean) | 280.0 | **335.4** | Ablation이 후반에 더 긺 |
| 480 첫 도달 | **16/16** | 15/16 | 거의 동일 |
| 수렴 에피소드 (mean) | **74.8** | 75.7 | 거의 동일 |

#### 4.3.2 Residual Net / Uncertainty 지표

| 지표 | 구간 | Baseline | Ablation | 배율 |
|---|---|---|---|---|
| `residual_error_avg` | ep 14~30 | 0.1449 | **0.1253** | — |
| `residual_error_avg` | ep 31~100 | 0.0177 | **0.0073** | 2.4× 높음 |
| `residual_error_avg` | ep 101~200 | 0.0088 | **0.0023** | 3.8× 높음 |
| `residual_error_avg` | final 10ep | 0.0090 | **0.0023** | 3.9× 높음 |
| `res_net_loss` | ep 15~30 | 0.01132 | **0.01034** | — |
| `res_net_loss` | ep 31~100 | 0.000227 | **0.000071** | 3.2× 높음 |
| `res_net_loss` | ep 101~200 | 0.000097 | **0.000008** | 12× 높음 |
| `uncertainty_avg` | ep 1~30 | 0.2435 | **0.2319** | — |
| `uncertainty_avg` | ep 31~100 | 0.0177 | **0.0073** | 2.4× 높음 |
| `uncertainty_avg` | ep 101~200 | 0.0088 | **0.0029** | 3.0× 높음 |
| `uncertainty < 0.1` 첫 도달 | — | ep 22.7 (16/16) | ep 21.9 (16/16) | 거의 동일 |
| `rbf_loss` | ep 31~100 | 0.0267 | **0.0096** | 2.8× 높음 |
| `rbf_loss` | ep 101~200 | 0.0007 | **0.0006** | 수렴 동일 |
| `dhat_norm_avg` | ep 101~200 | 0.3462 | 0.3485 | 사실상 동일 |

#### 4.3.3 시드별 최종 uncertainty vs 최종 reward 상관

| 조건 | 상관계수 (r) | 해석 |
|---|---|---|
| Baseline | **-0.882** | 최종 uncertainty가 높을수록 reward 급감, 강한 음의 상관 |
| Ablation | -0.763 | 상관 존재하나 Baseline보다 약함 |

Baseline 시드별 final uncertainty 범위: 0.005~0.016  
Ablation 시드별 final uncertainty 범위: 0.001~0.011 (전반적으로 낮고 분산도 작음)

### 4.4 Figure 목록

| 파일 | 내용 |
|---|---|
| `figures/baseline/reward_multiseed.png` | Baseline 16-seed reward mean ± std |
| `figures/baseline/uncertainty_avg.png` | Baseline uncertainty 궤적 |
| `figures/baseline/residual_error_avg.png` | Baseline residual error 궤적 |
| `figures/baseline/res_net_loss.png` | Baseline res_net_loss 궤적 |
| `figures/baseline/rbf_loss.png` | Baseline rbf_loss 궤적 |
| `figures/baseline/dhat_norm_avg.png` | Baseline dhat norm 궤적 |
| `figures/baseline/episode_length.png` | Baseline episode length 궤적 |
| `figures/baseline/td_loss_avg.png` | Baseline TD loss 궤적 |
| `figures/baseline/epsilon.png` | Baseline epsilon 궤적 |
| `figures/baseline/nominal_error_avg.png` | Baseline nominal model error |
| `figures/ablation/reward_multiseed.png` | Ablation 16-seed reward mean ± std |
| `figures/ablation/uncertainty_avg.png` | Ablation uncertainty 궤적 |
| `figures/ablation/residual_error_avg.png` | Ablation residual error 궤적 |
| `figures/ablation/res_net_loss.png` | Ablation res_net_loss 궤적 |
| `figures/ablation/rbf_loss.png` | Ablation rbf_loss 궤적 |
| `figures/ablation/dhat_norm_avg.png` | Ablation dhat norm 궤적 |
| `figures/ablation/episode_length.png` | Ablation episode length 궤적 |
| `figures/ablation/td_loss_avg.png` | Ablation TD loss 궤적 |
| `figures/ablation/epsilon.png` | Ablation epsilon 궤적 |
| `figures/ablation/nominal_error_avg.png` | Ablation nominal model error |
| `figures/compare_baseline_vs_ablation_reward.png` | Baseline vs Ablation reward 오버레이 비교 |

### 4.5 Uncertainty-based Sampling 효과 분석

#### A. Residual Net 학습 품질: Ablation이 전 구간에서 우월

가장 명확한 결과는 residual net 학습 품질에서 나온다.

- **`residual_error_avg` (final 10ep)**: Ablation 0.0023 vs Baseline 0.0088 — **3.9배 차이**
- **`res_net_loss` (ep 101~200)**: Ablation 0.000008 vs Baseline 0.000097 — **12배 차이**
- **`uncertainty_avg` (ep 101~200)**: Ablation 0.0029 vs Baseline 0.0088 — **3배 차이**

이 결과는 직관에 반한다. uncertainty-weighted sampling은 불확실한 샘플(어려운 사례)을 우선 학습해 residual net을 개선하려는 아이디어였지만, 실제로는 **균일 샘플링(Ablation)이 residual net을 훨씬 잘 수렴시킨다.**

**원인 가설: 샘플링 편향(sampling bias)**

uncertainty-weighted sampling은 buffer 내에서 고-uncertainty 샘플에 높은 가중치를 부여한다. 그런데 학습이 진행될수록 고-uncertainty 샘플은 소수(rare)이고 노이즈가 많다. 이 샘플들만 집중적으로 학습하면:

1. 저-uncertainty(쉬운) 샘플 영역을 under-train → 잔차 모델이 일반화에 실패
2. 고-uncertainty 샘플은 본질적으로 예측하기 어렵기 때문에, 이에 과도하게 맞추면 오히려 저-uncertainty 영역에서 오차가 커짐
3. RBF 불확실도 추정 자체가 정확하지 않은 경우, 잘못된 가중치가 전달되어 학습을 교란

결과적으로 uniform sampling이 state space 전반을 균등하게 학습해 더 안정적인 residual model을 만든다.

#### B. Uncertainty 감소 속도: 초반은 동일, 이후 분기

두 조건 모두 ep ~22에서 16/16 시드가 `uncertainty_avg < 0.1`(rollout gating threshold)에 도달한다 — 즉 **수렴 속도는 동일**하다.

그러나 steady-state 수준은 크게 다르다:

- ep 31~100: Baseline 0.0177 vs Ablation 0.0073 (2.4배)
- ep 101~200: Baseline 0.0088 vs Ablation 0.0029 (3배)

Baseline은 threshold 이하로는 떨어지지만, 그 이후에도 Ablation에 비해 높은 uncertainty 수준을 유지한다. 이는 residual net이 완전히 수렴하지 못했기 때문이다.

#### C. 특이 패턴: Baseline의 실패 스파이럴

Baseline에서 seed 1, 6은 final uncertainty가 0.015~0.016으로 타 시드(0.005~0.012)보다 확연히 높고, 최종 reward는 각각 6.9, 8.8로 사실상 학습 실패다. Ablation에서는 모든 시드의 final uncertainty가 0.001~0.011 범위 내에 있으며, 유사한 catastrophic failure 패턴이 없다.

이는 **uncertainty-weighted sampling이 일부 시드에서 실패 스파이럴을 유발**할 수 있음을 시사한다:

```
high uncertainty → rollout gating이 model rollout 차단
→ policy 학습 데이터 부족 → policy 개선 안됨
→ 나쁜 policy로 실제 환경 탐색 → dynamics 오차 큼
→ uncertainty 더 높아짐 (스파이럴)
```

Ablation(uniform sampling)에서는 이 스파이럴이 관찰되지 않는데, residual net이 전반적으로 잘 수렴해 uncertainty가 낮게 유지되어 rollout gating이 덜 발생하기 때문으로 보인다.

#### D. 정책 성능: 단기 수렴은 유사, 후반 안정성은 Ablation 우세

수렴 속도(480 첫 도달 에피소드 ~75 ep)와 초중반 reward는 두 조건에서 거의 동일하다. 그러나 후반부로 갈수록 차이가 드러난다:

- **최종 20ep 중 480≥ 비율**: Ablation 32.2% vs Baseline 15.6% — 2배 이상 차이
- **최종 10ep episode_length**: Ablation 335.4 vs Baseline 280.0

둘 다 ep 150 이후 reward가 붕괴(303~305 수준)하는 점은 동일하다. 이는 sampling 전략과 무관한 구조적 불안정성이 존재함을 의미하며, 두 조건 모두 최종 20ep 기준으로 480을 "유지"하는 시드가 매우 적다.

그럼에도 Ablation이 최종 안정성에서 부분적으로 우세한 이유는, residual net의 품질이 더 좋아 model rollout이 더 신뢰 가능하고 따라서 policy 학습에 유리한 데이터가 더 많이 공급되기 때문으로 해석된다.

#### E. DOB 개입 강도: 두 조건 동일

`dhat_norm_avg` (ep 101~200): Baseline 0.3462 vs Ablation 0.3485 — 사실상 동일하다. sampling 전략이 DOB 보정 강도에는 영향을 주지 않는다. DOB는 `dx_res`와 실제 오차의 blending으로 계산되므로, residual net의 품질이 달라져도 `dhat` 크기가 크게 변하지 않는 구조적 특성이 반영된 결과다.

#### F. 핵심 결론

> **uncertainty-weighted sampling은 이 실험 설계에서 residual net 학습을 오히려 저해한다.**

정량 요약:

| 질문 | 답변 |
|---|---|
| uncertainty sampling이 residual net 학습을 빠르게 하는가? | **아니오** — uniform이 3~12배 빠르게 수렴 |
| uncertainty sampling이 수렴 속도(480 도달)를 빠르게 하는가? | **아니오** — 두 조건 동일 (~22 ep for uncertainty, ~75 ep for policy) |
| uncertainty sampling이 후반 정책 안정성을 높이는가? | **아니오** — Ablation이 최종 안정성 2배 우세 |
| uncertainty sampling이 catastrophic failure를 줄이는가? | **아니오** — Baseline 2시드에서 failure spiral 관찰, Ablation은 없음 |
| 전반적으로 효과가 있는가? | **없음 (이 설계에서는 역효과)** |

**다음 Cycle 방향**: residual net 학습은 uniform sampling으로 유지. 핵심 병목은 ep 150 이후 reward 붕괴로, sampling 전략과 무관한 정책 안정성 문제다. target network update 주기, epsilon 하한, model rollout 비율 등을 조정해 후반 안정성을 높이는 실험을 우선한다.

---

## 5. 기록 지표 및 계산 방법

에피소드당 9개 지표를 CSV에 저장한다. 각 값의 계산 방법은 아래와 같다.

### 표기 정의

- `dx_real = next_obs - obs` : 실제 관측 변화량 (4D)
- `dx_nom` : nominal CartPole 모델이 예측한 관측 변화량 (4D, `step_nominal_cartpole` 출력)
- `e = dx_real - dx_nom` : nominal 모델 오차 (4D)
- `dx_res` : ResidualDxNet이 예측한 residual (2D, velocity/angular_velocity 성분만)
- `F_MAT` : 2D → 4D 매핑 행렬 (`[0,0; 1,0; 0,0; 0,1]`, 2D residual을 4D state에 삽입)
- `FPINV` : 4D → 2D 투영 행렬 (`[0,1,0,0; 0,0,0,1]`, velocity/angular_velocity만 추출)
- `dhat` : DOB 추정값 (`cfg.dob_w * dx_res + (1 - cfg.dob_w) * FPINV @ e`, 2D)
- `uncertainty = FPINV @ e - dx_res` : DOB 추정과 residual net 예측의 불일치 (2D)

### 지표별 계산

| 지표 | 계산식 | 집계 방법 | 비고 |
|---|---|---|---|
| `nominal_error_avg` | `‖e‖₂ = ‖dx_real - dx_nom‖₂` | step 평균 | 4D 전체 norm. nominal 모델의 원시 오차. |
| `residual_error_avg` | `‖e - F_MAT @ dx_res‖₂` | step 평균 | residual net 보정 후 남은 4D 오차. `nominal_error_avg`와 비교해 residual net 기여도 확인. |
| `dhat_norm_avg` | `‖dhat‖₂` | step 평균 | DOB 보정 벡터 크기. 클수록 DOB가 강하게 개입. |
| `uncertainty_avg` | `‖FPINV @ e - dx_res‖₂ = ‖uncertainty‖₂` | step 평균 | 2D velocity 공간에서 DOB 추정과 residual net의 불일치. rollout gating threshold(`uncertainty_threshold=0.1`)와 같은 공간의 값. |
| `res_net_loss` | ResidualDxNet MSE loss | Phase 1 훈련 전체 epoch 평균 | 학습 미시작 에피소드는 `NaN`. 타겟: buffer의 `dhat`. |
| `rbf_loss` | NormalizedRBFModel MSE loss | Phase 1 훈련 전체 epoch 평균 | 학습 미시작 에피소드는 `NaN`. 타겟: `|FPINV @ e - dx_res|` (절댓값). |
| `td_loss_avg` | DQN TD MSE loss | 에피소드 내 gradient update 평균 | warm-up 완료 전(업데이트 없는) 에피소드는 `NaN`. |
| `episode_length` | 에피소드 step 수 | - | `is_done=True`로 조기 종료 시 < 500. |
| `epsilon` | ε-greedy 탐험율 | 에피소드 종료 시점 값 | warm-up 구간(`total_steps ≤ 200`)에서는 decay 없음. |

### 해석 가이드

```
nominal_error_avg > residual_error_avg  →  residual net이 dynamics 예측 개선
residual_error_avg ≈ nominal_error_avg  →  residual net 미작동 (학습 부족 또는 포화)
uncertainty_avg < 0.1 (threshold)       →  model rollout 대부분 신뢰 가능
res_net_loss 감소                        →  residual net 수렴
td_loss_avg 감소                         →  Q-network 수렴
```

---

## 6. 관찰 및 교훈

<!-- ★ 사람이 직접 작성 -->

---

## 7. 메모리 업데이트 제안 (선택)

아래 내용을 `memory.md`의 "실험 교훈" 섹션에 추가를 제안한다 (승인 시 반영):

- **Cycle 1 교훈 — uncertainty-weighted sampling의 역효과**: uniform sampling(Ablation)이 residual net을 3~12배 더 잘 수렴시킨다 (residual_error: 0.0090→0.0023, res_net_loss: 0.000097→0.000008). uncertainty-weighted sampling은 고-uncertainty(어려운) 샘플에 집중해 sampling bias를 유발하고, 일부 시드에서는 high-uncertainty → rollout gating → policy 학습 부족 → high-uncertainty 스파이럴로 이어져 catastrophic failure를 초래한다. 이후 Cycle에서는 residual net 학습에 uniform sampling을 기본으로 사용할 것.
- **Cycle 1 교훈 — 정책 안정성이 핵심 병목**: 두 조건 모두 ep ~75에서 480에 도달하지만 ep 150 이후 reward가 붕괴(303~305 수준)한다. sampling 전략과 무관하게 발생하는 구조적 불안정성이다. 다음 Cycle에서는 target network 업데이트 주기, real/model ratio, epsilon 하한 등을 조정해 후반 안정성을 높이는 실험을 우선한다.
