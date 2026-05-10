# Cycle 3 실험 보고서

> 생성일: 2026-05-10
> 시작점: Cycle_2

## 1. 변경점 요약

Cycle 2 분석에서 RBF uncertainty gating이 처음부터 완전히 비활성화 상태임을 확인했다.
근본 원인은 두 가지: (1) 28D 공간에서 `rbf_width=0.25`는 수학적으로 무의미하여 모든 phi ≈ 0,
(2) 학습 타겟이 "현재 res_net이 buffer 데이터를 얼마나 잘 맞추는가"이므로 res_net이 학습될수록 타겟 → 0.
이를 수정하여 RBF가 실제로 위치 구분을 할 수 있도록 한다.

## 2. 변경 상세

| 파일 | 변경 내용 |
|---|---|
| `config.py` | `rbf_width`: 0.25 → 3.65 (median heuristic), `rbf_initial_value`: 5.0 → 0.5, `lr_rbf`: 0.5 → 0.01 |
| `model_learning.py` | `train_uncertainty_rbf`: `res_net` 인자 제거, 타겟을 `real_buffer.uncertainty`(수집 당시 저장값)로 변경 |
| `normalized_rbf.py` | phi 정규화: `sum` → `max` (모르는 영역에서 자연스럽게 낮은 출력) |
| `trainer.py` | `train_uncertainty_rbf` 호출 시 `res_net` 인자 제거 |

### rbf_width 계산 근거
28D 정규화 공간 `[-1, 1]^28`에서 임의의 두 점 사이 예상 거리:
```
E[dist_sq] ≈ 18.7  (median ≈ 18.5)
```
Median heuristic으로 계산한 적절한 width:
```
width = sqrt(median_dist_sq / (2 * ln(2))) = sqrt(18.5 / 1.386) ≈ 3.65
```

### phi 정규화 변경 근거
- **기존(sum)**: phi가 모두 ≈ 0이어도 정규화가 uniform 분포를 만들어 "모르는 상태" 신호 소실
- **수정(max)**: 가장 가까운 center가 기준 → 멀리 떨어진 상태는 max phi 자체가 작아져 출력이 작아짐

## 3. 하이퍼파라미터

| 파라미터 | Cycle 2 | Cycle 3 |
|---|---|---|
| `rbf_width` | 0.25 | 3.65 |
| `rbf_initial_value` | 5.0 | 0.5 |
| `lr_rbf` | 0.5 | 0.01 |
| 나머지 | 동일 | 동일 |

## 4. 학습 결과

<!-- ★ 사람이 직접 작성 -->

## 5. 관찰 및 교훈

<!-- ★ 사람이 직접 작성 -->

### 검증 기준

| 지표 | 기대 동작 |
|---|---|
| `rollout_uncert_avg` | 0이 아닌 값 (0.1~0.5 범위), 학습 초반에 높고 후반에 낮아져야 함 |
| `buffer_uncert_avg` vs `rollout_uncert_avg` | 비슷한 스케일이어야 함 (Cycle 2: 0.4 vs ~0) |
| `rbf_calib_ratio` | 0.8~1.2 범위 |
| `rbf_calib_corr` | 0.5 이상 |

## 6. 메모리 업데이트 제안 (선택)

<!-- ★ 학습 완료 후 작성 -->
