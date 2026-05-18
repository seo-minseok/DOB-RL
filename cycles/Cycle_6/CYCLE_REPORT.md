# Cycle 6 실험 보고서

> 생성일: 2026-05-18
> 시작점: Cycle_5

## 1. 변경점 요약

Cycle 5 대비 두 가지 변경:
1. `model_learning.py` — `train_residual_dx_model_dob()`의 uncertainty-weighted sampling 제거, 균등 랜덤 샘플링으로 변경
2. `rollout.py` — RBF uncertainty 기반 rollout truncation 제거, 지정 horizon까지 무조건 rollout

## 2. 변경 상세

### `dob_mbrl/training/model_learning.py`
- **제거**: `uncert_mag` → `weights` → `probs` 계산 및 `np.random.choice(..., p=probs)` 호출
- **변경**: `np.random.randint(0, valid_len, size=mini_batch_size)` 균등 샘플링으로 대체
- 반환값 구조(`loss_avg`, `sampled_uncert_avg`)는 동일하게 유지 (sampled_uncert_avg는 이제 단순 평균)

### `dob_mbrl/training/rollout.py`
- **제거**: `is_reliable` 마스크, `pass_rate_vals`, uncertainty threshold 기반 조기 종료 로직
- **제거**: `n_reliable == 0` break 조건, `alive_mask[alive_idx[~is_reliable]] = False`
- **유지**: RBF uncertainty 계산 (로깅 목적으로만)
- **변경**: 모든 alive trajectory에 대해 `predict_next_obs_dob` 수행 (신뢰도 필터링 없음)
- `rollout_pass_rate`는 항상 `1.0` 반환 (truncation 없음을 명시)

## 3. 하이퍼파라미터

Cycle 5와 동일 (config.py 변경 없음).

## 4. 학습 결과

<!-- ★ 학습 완료 후 작성 -->

## 5. 관찰 및 교훈

<!-- ★ 사람이 직접 작성 -->

## 6. 메모리 업데이트 제안 (선택)

<!-- 에이전트가 필요 시 자동 생성 -->
