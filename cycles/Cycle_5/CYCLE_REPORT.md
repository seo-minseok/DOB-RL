# Cycle 5 실험 보고서

> 생성일: 2026-05-12
> 시작점: Cycle_4

## 1. 변경점 요약

Cycle 4의 MBRL 파이프라인에서 발견된 4가지 버그 수정.

| # | 파일 | 변경 내용 |
|---|---|---|
| Fix 1 | `utils/buffer.py`, `training/trainer.py` | `model_buffer` 에피소드마다 초기화 (`reset()`) |
| Fix 2 | `training/rollout.py` | rollout h==0 무조건 통과 제거 — 모든 step uncertainty gating 적용 |
| Fix 3 | `training/model_learning.py`, `training/trainer.py` | RBF 학습 타깃을 현재 res_net 기준으로 재계산 |
| Fix 4 | `envs/bipedalwalker_utils.py`, `training/rollout.py` | synthetic reward를 Gymnasium 공식 수식으로 정교화 |

## 2. 변경 상세

### Fix 1 — model_buffer 주기적 초기화

- **문제**: `model_buffer`가 학습 내내 누적되어 오래된 synthetic transition이 critic 학습을 오염.
- **수정**: `ReplayBufferDOB.reset()` 메서드 추가. `trainer.py` Phase 1에서 `generate_samples_dob` 호출 직전에 `model_buffer.reset()` 호출.
- **효과**: 매 에피소드 rollout은 오직 현재 모델 기준의 fresh synthetic data만 포함.

### Fix 2 — rollout 첫 step 무조건 통과 제거

- **문제**: `h==0`에서 `is_reliable[:] = True`로 강제 통과시켜 uncertainty가 높은 상태에서도 1-step synthetic transition이 무조건 삽입됨.
- **수정**: `if h == 0: is_reliable[:] = True` 블록 제거. `pass_rate_vals`도 h==0부터 수집.
- **효과**: uncertainty threshold를 통과하지 못하면 첫 step부터 rollout을 중단 — 품질 낮은 transition 차단.
- **주의**: 학습 초기(RBF 미학습 시) rollout 통과율이 0이 될 수 있어 model_buffer가 비어있을 수 있음. `sample_mixed_minibatch`는 이 경우 `model_trained_at_least_once` 플래그로 real_buffer만 사용하므로 안전.

### Fix 3 — RBF 학습 타깃 복원 (현재 res_net 기준)

- **문제**: Cycle 4에서 RBF 타깃을 buffer 저장 시점의 uncertainty로 변경했으나, res_net이 업데이트될수록 저장된 uncertainty가 stale해져 RBF가 구식 분포를 학습.
- **수정**: `train_uncertainty_rbf`에 `res_net` 인자 추가. 매 배치마다 `FPINV @ (dx_real - dx_nom) - res_net(obs, act)`를 현재 res_net으로 재계산해 타깃으로 사용.
- **시그니처 변경**: `train_uncertainty_rbf(uncert_model, optimizer, real_buffer, res_net, batch_size, epochs)`.

### Fix 4 — BipedalWalker synthetic reward 정교화

- **문제**: 기존 `reward = 0.72 * vel_x` (next_obs만 사용)는 각도 변화 패널티, 모터 비용 누락.
- **수정**: `reward_is_done_function(obs, action, next_obs)` 시그니처로 변경.
  ```
  Δshaping = (130/30) * (obs[2]/6) - 5.0 * (|next_obs[0]| - |obs[0]|)
  motor_cost = 0.00035 * 80 * Σ clip(|action|, 0, 1)
  reward = Δshaping - motor_cost  (game_over → -100)
  ```
- rollout.py 호출부도 `reward_is_done_function(rel_obs, rel_act, rel_next)`로 변경.

## 3. 하이퍼파라미터

Cycle 4와 동일. `config.py` 수정 없음.

## 4. 학습 결과

<!-- ★ 사람이 직접 작성 -->

## 5. 관찰 및 교훈

<!-- ★ 사람이 직접 작성 -->

## 6. 메모리 업데이트 제안 (선택)

- **제거 제안**: `memory.md` 항목 "RBF rollout 첫 번째 step (h==0)에서는 항상 신뢰 가능으로 처리 (is_reliable[:] = True)" — Fix 2로 해당 동작 제거됨.
