# Cycle_1-mbpo 실험 보고서

> 생성일: 2026-06-07
> 최종 수정: 2026-06-09
> 시작점: 신규 (mbpo/ 레퍼런스 + Cycle_1-dob-rl 코드 스타일)

## 1. 변경점 요약

- Cycle_1-dob-rl (DOB-MBRL)에서 DOB / RBF / Residual Net 완전 제거
- MBPO 원본(TF) 논문 하이퍼파라미터를 PyTorch로 재현
- Probabilistic Ensemble(BNN) 7개 네트워크, 5 elites, 4×200 Swish 구조
- 원본 MBPO 대비 수정된 SAC 설정 적용 (init_alpha, target_entropy, critic loss)

---

## 2. Cycle_1-dob-rl 대비 변경 상세

| 항목 | Cycle_1-dob-rl (DOB-MBRL) | Cycle_1-mbpo (MBPO) |
|---|---|---|
| Dynamics Model | Single Residual DxNet (2×256 ReLU) | Probabilistic Ensemble (4×200 Swish, 7 nets, 5 elites) |
| 출력 | Δobs (11D, deterministic) | mean + log_var (12D = Δobs+Δrew, probabilistic) |
| DOB 보정 | ✓ | ✗ |
| RBF Uncertainty | ✓ (선택) | ✗ |
| Model Optimizer | Adam (lr=1e-3) | Adam (lr=1e-3) |
| Gradient steps/step | 20 | 20 |
| real_ratio | 0.05 | 0.05 |
| rollout_horizon | 1→12 schedule | 1→12 schedule |
| rollout_batch | 100,000 | 100,000 |
| Critic loss | 0.5 × MSE | 0.5 × MSE |

---

## 3. 구현 상세

### Ensemble Model 구조
- 입력: `(obs, act)` = (11 + 3 = 14D)
- 출력: `mean, log_var` 각 12D (Δobs 11D + reward 1D)
- Hidden: 4층 × 200 nodes, Swish activation
- 네트워크 수: 7개, Elite: 5개 (validation NLL 기준 선택)
- Log-variance 범위: `[-10, 0.5]` (softplus clamp)

### SAC 설정 (원본 MBPO 기준)
- `init_alpha = 1.0` (log_alpha = 0.0)
- `target_entropy = -1.0` (Hopper 원본 설정, auto=-3 아님)
- Critic loss = `0.5 × MSE` (원본 TF MBPO)
- Gradient clipping 없음

### Model 학습
- Holdout ratio: 0.2 (80% train / 20% validation)
- Early stopping: `max_epochs_since_update = 5` 이후 개선 없으면 중단
- Elite 선택: holdout NLL 기준 상위 5개

---

## 4. 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---|---|---|
| total_steps | 125,000 | |
| max_steps_per_ep | 1,000 | |
| warm_start_steps | 5,000 | |
| epoch_length | 1,000 | |
| lr (SAC actor/critic/alpha) | 3e-4 | Adam |
| model_lr | 1e-3 | Adam |
| discount_factor | 0.99 | |
| tau | 5e-3 | soft update |
| init_alpha | 1.0 | log_alpha=0.0 |
| target_entropy | -1.0 | Hopper 원본 MBPO |
| n_train_repeat | 20 | gradient steps per env step |
| real_ratio | 0.05 | 5% real, 95% model |
| rollout_interval | 250 | steps |
| rollout_schedule | [20, 150, 1, 15] | epoch 기준 horizon 1→12 (125k 내) |
| rollout_batch_size | 100,000 | |
| model_retain_epochs | 1 | |
| mini_batch_size | 256 | |
| buffer_size | 1,000,000 | |
| num_networks | 7 | |
| num_elites | 5 | |
| hidden_dim | 200 | 4-layer |
| holdout_ratio | 0.2 | |
| max_epochs_since_update | 5 | early stopping |

---

## 5. 학습 결과 (seeds 1-8, Hopper-v5)

| Seed | Best eval_reward |
|---|---|
| 1 | 2,604 |
| 2 | 3,249 |
| 3 | 3,208 |
| 4 | 3,293 |
| 5 | 3,429 |
| 6 | 1,768 |
| 7 | 3,342 |
| 8 | 3,490 |
| **평균** | **3,047.9 ± 584.2** |

### 비교

| 항목 | Cycle_1-mbpo | Cycle_1-dob-rl (Adam) | SAC (baseline) |
|---|---|---|---|
| Mean best | 3,047.9 ± 584.2 | 3,230.6 ± 157.7 | ~3,300+ |
| 최저 seed | 1,768 (seed 6) | 2,890 (seed 5) | — |
| 분산 | 높음 | 낮음 | — |

---

## 6. 관찰 및 교훈

- 원본 MBPO 논문의 target_entropy=-1, init_alpha=1.0 설정이 성능에 결정적 영향 (초기에 -3.0으로 설정 시 return 350~470에서 정체)
- 앙상블 모델의 seed 간 분산(std=584)이 Cycle_1-dob-rl(std=158)보다 높음 — 앙상블 자체의 확률론적 특성 및 elite 선택 결과에 따른 변동
- Seed 6이 유독 낮은 이유 불명확 — 앙상블 초기화 운에 따른 가능성 있음
- eval_interval=50 에피소드 기준이 후반 긴 에피소드에서 eval 포인트 부족 유발 (Cycle_1-dob-rl과 동일한 문제)

---

## 7. 다음 Cycle 제안

- `eval_interval`을 step 기준으로 변경
- Seed 분산 감소를 위한 앙상블 초기화 안정화 방법 검토
- Cycle_1-dob-rl과 MBPO를 결합한 DOB+Ensemble 구조 실험
