# Cycle 1 실험 보고서

> 생성일: 2026-04-10
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
| Final 10-ep avg — mean ± std | **252.38 ± 148.21** |
| Final 10-ep avg — max / min | 470.61 / 6.94 |
| Seeds ≥ 480 (최종 유지) | **0 / 16** |
| Seeds < 50 (catastrophic failure) | **3 / 16** (Seed 1: 6.94, Seed 2: 8.59, Seed 6: 8.79) |
| 전체 에피소드 reward — mean ± std | 307.81 ± 189.84 |
| 총 환경 스텝 — mean (min / max) | 65,893 (47,123 / 78,399) |
| 480 첫 도달 시드 수 | **16 / 16** |
| 480 첫 도달 에피소드 — mean / median | 50.9 / 51.0 |

시드별 Final 10-ep avg:

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Reward | 6.9 | 8.6 | 432.9 | 297.6 | 250.3 | 8.8 | 470.6 | 278.2 | 215.3 | 233.1 | 217.3 | 223.4 | 228.5 | 320.4 | 433.9 | 412.4 |

### 4.2 Ablation (uniform sampling)

| 지표 | 값 |
|---|---|
| 시드 수 | 16 |
| 에피소드 수 | 200 |
| Final 10-ep avg — mean ± std | **315.66 ± 184.05** |
| Final 10-ep avg — max / min | 490.90 / -2.37 |
| Seeds ≥ 480 (최종 유지) | **3 / 16** |
| Seeds < 50 (catastrophic failure) | **2 / 16** (Seed 6: 4.27, Seed 14: -2.37) |
| 전체 에피소드 reward — mean ± std | 313.11 ± 190.66 |
| 총 환경 스텝 — mean (min / max) | 66,670 (41,721 / 79,360) |
| 480 첫 도달 시드 수 | **16 / 16** |
| 480 첫 도달 에피소드 — mean / median | 37.4 / 35.0 |

시드별 Final 10-ep avg:

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Reward | 445.0 | 490.9 | 461.1 | 413.3 | 481.5 | 4.3 | 345.2 | 489.0 | 478.8 | 417.9 | 52.4 | 67.5 | 300.8 | -2.4 | 352.3 | 252.9 |

### 4.3 Baseline vs Ablation 비교 요약

#### 4.3.1 Reward 지표

| 지표 | Baseline | Ablation | 해석 |
|---|---|---|---|
| Final 10-ep avg (mean) | 252.38 | **315.66** | Ablation이 63포인트 우세 |
| Final 10-ep avg (std) | 148.21 | 184.05 | Ablation이 분산 더 큼 |
| Seeds ≥ 480 (최종) | 0 / 16 | **3 / 16** | Ablation만 480 유지 존재 |
| Seeds < 50 (catastrophic) | 3 / 16 | 2 / 16 | Baseline이 실패 시드 더 많음 |
| 전체 ep reward (mean) | 307.81 | **313.11** | 거의 동일 |
| ep 1~50 reward mean | 117.15 | **153.25** | Ablation이 초반 더 빠름 |
| ep 51~100 reward mean | 393.16 | **402.78** | Ablation이 수렴 구간 소폭 우세 |
| ep 101~150 reward mean | **417.20** | 391.35 | Baseline이 중반 이후 역전 |
| ep 151~200 reward mean | 303.73 | 305.04 | 후반 붕괴 동일 수준 |
| 최종 20ep 중 480≥ 비율 | 15.6% | **32.2%** | Ablation이 2배 이상 높음 |
| 최종 10ep episode_length (mean) | 280.02 | **335.39** | Ablation이 후반에 더 긺 |
| 480 첫 도달 | 16/16 | 16/16 | 동일 |
| 480 첫 도달 에피소드 (mean) | 50.9 | **37.4** | Ablation이 13ep 빠름 |
| 480 첫 도달 에피소드 (median) | 51.0 | **35.0** | Ablation이 더 일관되게 빠름 |

#### 4.3.2 Residual Net / Uncertainty 지표

| 지표 | 구간 | Baseline | Ablation | 배율 |
|---|---|---|---|---|
| `residual_error_avg` | ep 14~30 | 0.1449 | **0.1253** | — |
| `residual_error_avg` | ep 31~100 | 0.01774 | **0.00727** | 2.4× |
| `residual_error_avg` | ep 101~200 | 0.00883 | **0.00292** | 3.0× |
| `residual_error_avg` | final 10ep | 0.00901 | **0.00235** | 3.8× |
| `res_net_loss` | ep 15~30 | 0.011316 | **0.010344** | — |
| `res_net_loss` | ep 31~100 | 0.000227 | **0.000071** | 3.2× |
| `res_net_loss` | ep 101~200 | 0.000097 | **0.000008** | 11.6× |
| `uncertainty_avg` | ep 1~30 | 0.2435 | **0.2319** | — |
| `uncertainty_avg` | ep 31~100 | 0.01774 | **0.00727** | 2.4× |
| `uncertainty_avg` | ep 101~200 | 0.00883 | **0.00292** | 3.0× |
| `uncertainty < 0.1` 첫 도달 | — | ep 22.7 (16/16) | ep 21.9 (16/16) | 거의 동일 |
| `rbf_loss` | ep 31~100 | 0.02670 | **0.00957** | 2.8× |
| `rbf_loss` | ep 101~200 | 0.000676 | **0.000595** | 수렴 동일 |
| `dhat_norm_avg` | ep 101~200 | 0.3462 | 0.3485 | 사실상 동일 |
| `td_loss_avg` | ep 51~100 | **1.673** | 2.332 | Ablation이 1.4× 높음 |
| `td_loss_avg` | ep 101~200 | **7.999** | 27.211 | Ablation이 3.4× 높음 |

#### 4.3.3 시드별 최종 uncertainty vs 최종 reward 상관

| 조건 | 상관계수 (r) | 해석 |
|---|---|---|
| Baseline | **-0.882** | 최종 uncertainty가 높을수록 reward 급감, 강한 음의 상관 |
| Ablation | -0.763 | 상관 존재하나 Baseline보다 약함 |

Baseline 시드별 final uncertainty 범위: 0.00464 ~ 0.01621
Ablation 시드별 final uncertainty 범위: 0.00063 ~ 0.01051

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

---

## 5. Uniform vs Uncertainty-weighted Sampling 분석

### A. 핵심 전제: 두 방식의 차이

**Uncertainty-weighted sampling (Baseline)**: Replay buffer에서 샘플을 추출할 때 각 샘플의 `uncertainty`(= `‖FPINV @ e - dx_res‖₂`)를 가중치로 사용한다. 불확실도가 높은 transition을 더 빈번하게 학습에 사용해 residual net이 "어려운" 영역을 집중적으로 학습하게 유도하는 방식이다.

**Uniform sampling (Ablation)**: Replay buffer에서 균등 확률로 샘플을 추출한다. 각 transition이 동등하게 학습에 기여한다.

---

### B. Residual Net 학습 품질: Uniform이 압도적으로 우월

가장 명확한 결과는 residual net 학습 품질에서 나온다.

| 지표 | Baseline (weighted) | Ablation (uniform) | 배율 |
|---|---|---|---|
| `residual_error_avg` (final 10ep) | 0.00901 | **0.00235** | 3.8× |
| `res_net_loss` (ep 101~200) | 0.000097 | **0.000008** | 11.6× |
| `uncertainty_avg` (ep 101~200) | 0.00883 | **0.00292** | 3.0× |

이 결과는 직관에 반한다. Uncertainty-weighted sampling은 불확실한 샘플(어려운 사례)을 우선 학습해 residual net을 개선하려는 아이디어이지만, 실제로는 **균일 샘플링이 residual net을 훨씬 잘 수렴시킨다.**

**원인 분석 — 샘플링 편향(sampling bias)**

학습이 진행될수록 buffer 내 고-uncertainty 샘플은 감소하고 노이즈가 많다. Uncertainty-weighted sampling은 이런 샘플에 집중하기 때문에:

1. **저-uncertainty(쉬운) 영역의 under-training**: 전체 state space에서 residual net이 일반화에 실패하고, 방문 빈도가 높은 "쉬운" 영역의 예측 정확도가 낮아진다.
2. **노이즈 과적합**: 고-uncertainty 샘플은 본질적으로 예측하기 어렵거나 이상치(outlier)일 가능성이 높다. 이를 집중 학습하면 안정적인 영역의 오차가 커진다.
3. **RBF uncertainty 추정 오차의 전파**: RBF의 불확실도 추정이 완전히 정확하지 않으면, 잘못된 가중치가 학습을 교란한다.

반면 Uniform sampling은 state space 전반을 균등하게 학습하므로 residual model의 전반적 일반화 성능이 높다.

---

### C. 초기 수렴 속도: Ablation이 더 빠름

Uncertainty < 0.1 첫 도달 시점은 두 조건 모두 ep ~22로 동일하지만, 480 reward 첫 도달에서는 명확한 차이가 난다.

| 지표 | Baseline | Ablation |
|---|---|---|
| `uncertainty < 0.1` 첫 도달 | ep 22.7 (16/16) | ep 21.9 (16/16) |
| 480 reward 첫 도달 (mean) | ep 50.9 (16/16) | **ep 37.4 (16/16)** |
| 480 reward 첫 도달 (median) | ep 51 | **ep 35** |

Rollout gating threshold를 넘는 시점(ep ~22)은 동일하지만, 이후 residual net이 더 빠르게 수렴하는 Ablation 조건에서 policy가 더 이른 에피소드부터 신뢰 가능한 model rollout을 활용해 학습할 수 있다. ep 1~50 구간 평균 reward(153.25 vs 117.15)도 Ablation의 초반 우세를 뒷받침한다.

---

### D. 중반 역전: Baseline이 ep 101~150에서 우세

흥미롭게도, ep 101~150 구간에서는 Baseline(417.20)이 Ablation(391.35)보다 reward가 높다. 이 구간은 두 조건 모두 policy가 어느 정도 수렴한 상태에서 안정적으로 운용되는 구간이다. Uncertainty-weighted sampling이 오히려 고-uncertainty 전이에 집중해 policy의 경계 조건 대응력을 높였을 가능성도 있으나, 이후 ep 151~200에서 두 조건 모두 비슷한 수준(303~305)으로 붕괴하므로 이 우세가 지속성 있는 장점이라고 보기 어렵다.

---

### E. 후반 안정성: Ablation 우세

ep 151 이후 두 조건 모두 reward가 붕괴하는 패턴은 동일하다. 그러나 최종 구간의 절대 수준에서 Ablation이 우세하다.

| 지표 | Baseline | Ablation |
|---|---|---|
| Final 10-ep mean | 252.38 | **315.66** |
| Seeds ≥ 480 (최종) | 0/16 | **3/16** |
| 최종 20ep 중 480≥ 비율 | 15.6% | **32.2%** |
| Final 10-ep episode_length | 280.02 | **335.39** |

Ablation의 후반 안정성 우위는 residual net 품질과 직결된다. residual_error와 uncertainty가 낮게 유지되면 model rollout의 신뢰도가 높아지고, 이는 Q-network 학습에 더 정확한 virtual transition을 공급하므로 policy 품질을 지지한다.

---

### F. Baseline의 실패 스파이럴 패턴

Baseline에서 seed 1, 2, 6은 final 10-ep reward가 6.94, 8.59, 8.79로 사실상 학습 실패다. 이 세 시드의 final uncertainty는 0.0152, 0.0143, 0.0162로, 정상 수렴 시드(0.005~0.010)보다 현저히 높다. 이는 아래 메커니즘으로 설명된다.

```
high residual_error (sampling bias)
→ uncertainty 높게 유지됨
→ rollout gating이 model rollout 차단
→ policy 학습 데이터 부족
→ 나쁜 policy로 실제 환경 탐색 → dynamics 오차 큼
→ uncertainty 더 높아짐 (스파이럴)
```

Baseline의 최종 uncertainty-reward 상관계수는 -0.882로, uncertainty가 높은 시드가 반드시 낮은 reward로 귀결되는 강한 패턴을 보인다. Ablation에서는 이 상관이 -0.763으로 약화되는데, residual net이 전반적으로 잘 수렴해 uncertainty가 낮게 유지되어 rollout gating이 덜 발동하기 때문이다.

단, Ablation도 실패가 없지는 않다. Seed 6과 14는 각각 4.27, -2.37로 catastrophic failure를 보이며, Seed 11과 12는 52.42, 67.47로 중간 수준의 실패를 기록한다. Ablation의 실패는 uncertainty 스파이럴보다는 다른 요인(초기 탐색 운, policy gradient 불안정성 등)에서 비롯된 것으로 보인다.

---

### G. TD loss: Ablation의 고-분산 Q-학습

예상치 못한 패턴이 td_loss에서 나타난다.

| 구간 | Baseline | Ablation |
|---|---|---|
| ep 51~100 | 1.673 | 2.332 |
| ep 101~200 | 7.999 | **27.211** |

Ablation의 TD loss가 Baseline 대비 ep 101~200에서 3.4배 높다. 이는 Ablation에서 model rollout이 더 다양한 state space를 커버하고, Q-target의 분산이 더 크기 때문으로 추정된다. 다만 이 높은 TD loss가 최종 policy 성능을 저해하지는 않았다 — Ablation이 최종 reward 측면에서 전반적으로 우세하기 때문이다. 단, 높은 TD loss는 Q-network의 수렴 안정성에 대한 우려를 남기며, 향후 target network 업데이트 주기나 학습률 조정을 검토할 이유가 된다.

---

### H. DOB 개입 강도: 두 조건 동일

`dhat_norm_avg` (ep 101~200): Baseline 0.3462 vs Ablation 0.3485 — 사실상 동일하다. sampling 전략이 DOB 보정 강도에는 영향을 주지 않는다. DOB는 `dx_res`와 실제 오차의 blending으로 계산되므로(`dob_w=0.1`), residual net의 품질이 달라져도 `dhat` 크기 자체는 nominal error에 의해 지배되어 큰 변화가 없다. 또한 nominal model의 오차(`nominal_error_avg`)도 두 조건에서 ep 101~200 기준 0.347 vs 0.349로 동일 — nominal model은 sampling 전략과 완전히 독립적으로 동작함이 확인된다.

---

### I. 핵심 결론

> **Uncertainty-weighted sampling은 이 실험 설계에서 residual net 학습을 저해하며, 초기 수렴 속도와 최종 안정성 모두 uniform sampling에 열세이다.**

정량 요약:

| 질문 | 답변 |
|---|---|
| uniform이 residual net을 더 잘 학습시키는가? | **그렇다** — 3~12배 빠르게 수렴 |
| uniform이 480 도달 속도를 빠르게 하는가? | **그렇다** — mean 50.9 → 37.4 ep (13ep 단축) |
| uniform이 후반 정책 안정성을 높이는가? | **그렇다** — 최종 20ep ≥480 비율 2배, seeds ≥480 0→3개 |
| uncertainty sampling이 catastrophic failure를 줄이는가? | **아니다** — Baseline 3 seeds < 10 reward, Ablation 2 seeds < 10 |
| 두 조건 모두 ep 150 이후 붕괴하는가? | **그렇다** — sampling 전략과 무관한 구조적 불안정성 |

**다음 Cycle 방향**: residual net 학습은 uniform sampling으로 확정. 핵심 병목은 ep 150 이후 reward 붕괴로, sampling 전략과 무관한 정책 안정성 문제이다. target network 업데이트 주기, epsilon 하한, model rollout 비율(real_ratio), TD loss 안정화를 조정해 후반 안정성을 높이는 실험을 우선한다.

---

## 6. 기록 지표 및 계산 방법

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

## 7. 관찰 및 교훈

<!-- ★ 사람이 직접 작성 -->

---

## 8. 메모리 업데이트 제안 (선택)

아래 내용을 `memory.md`의 "실험 교훈" 섹션에 추가를 제안한다 (승인 시 반영):

- **Cycle 1 교훈 — Uniform sampling의 우위**: Uniform sampling(Ablation)이 residual net을 Uncertainty-weighted sampling(Baseline) 대비 3~12배 더 잘 수렴시킨다 (residual_error final10: 0.00901→0.00235, res_net_loss ep101~200: 0.000097→0.000008). Uncertainty-weighted sampling은 고-uncertainty 샘플에 집중해 sampling bias를 유발하고, 일부 시드에서는 high-uncertainty → rollout gating → policy 학습 부족 → high-uncertainty 스파이럴로 이어져 catastrophic failure를 초래한다(Baseline 3 seeds < 10 reward). 이후 Cycle에서는 residual net 학습에 uniform sampling을 기본으로 사용할 것.
- **Cycle 1 교훈 — Uniform sampling의 초기 수렴 우위**: Ablation은 480 reward 첫 도달이 mean ep 37.4로, Baseline(50.9)보다 13ep 빠르다. residual net이 더 빠르게 수렴해 model rollout 신뢰도가 일찍 확보되기 때문이다.
- **Cycle 1 교훈 — 정책 안정성이 핵심 병목**: 두 조건 모두 ep 150 이후 reward가 붕괴한다(303~305 수준). sampling 전략과 무관하게 발생하는 구조적 불안정성이다. 다음 Cycle에서는 target network 업데이트 주기, real/model ratio, epsilon 하한 등을 조정해 후반 안정성을 높이는 실험을 우선한다.
- **Cycle 1 교훈 — Ablation의 고-분산 TD loss**: Ablation에서 ep 101~200 td_loss_avg가 27.2로 Baseline(8.0) 대비 3.4배 높다. 최종 성능에는 악영향이 없었으나, Q-network 수렴 안정성 관점에서 target network 업데이트 주기나 학습률 조정을 검토할 필요가 있다.
