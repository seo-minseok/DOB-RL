# RBF Uncertainty Model — 설계 분석 및 개선 방향

*최종 업데이트: 2026-06-10 (88,339 샘플 실험 기반 분석 추가)*

---

## 1. 현재 설정 (Cycle_1 기준)

| 항목 | 초기값 | 현재값 | 비고 |
|---|---|---|---|
| `num_rbf_centers` | 3000 | 3000 | |
| `rbf_width` | 0.8 | 0.6 | PHYS 수정 후 재분석 결과 |
| `rbf_initial_value` | 5.0 | 5.0 | |
| center 초기화 | PHYS uniform random | PHYS uniform random | data-centered 시도 후 복구 (섹션 5 참고) |
| 입력 공간 | 14D: obs(11D) + act(3D) | 동일 | 정규화 후 [-1, 1]^14 |
| `uncertainty_threshold` | — | 1.5 | |

**PHYS 범위 수정 내역** (88,339 샘플 실제 데이터 기반 + 20% margin):

| dim | 구 PHYS | 신 PHYS | 문제 |
|---|---|---|---|
| thigh | [-1.0, 1.0] | [-1.8, 0.4] | 실제 min=-1.49가 구 PHYS 초과 → clipping 발생 |
| leg | [-1.0, 1.0] | [-2.6, 0.6] | 실제 min=-2.17이 구 PHYS 심각하게 초과 |
| z | [-0.5, 2.5] | [0.5, 1.8] | 실제 범위 [0.70, 1.64], 구 PHYS의 22%만 사용 |
| root_angle | [-0.5, 0.5] | [-0.3, 0.3] | 실제 범위 [-0.20, 0.20], 40%만 사용 |

---

## 2. RBF 학습 메커니즘

```
query x (obs+act, 14D)
    ↓ 정규화 → [-1, 1]^14
    ↓
phi[i]   = exp(-||x - center_i||² / (2 × width²))   for i = 1..K
phi_norm = phi / (phi.sum() + 1e-8)                  # softmax 정규화
output   = phi_norm @ weights                         # weight 가중 평균
```

**핵심**: output은 모든 weights의 **phi_norm 가중 평균**이다.
특정 center가 지배적으로 활성화(phi_norm_i ≈ 1)되어야 local 학습 및 OOD 탐지가 가능하다.

---

## 3. 근본 원인: 14D에서의 소프트맥스 정규화 문제

### 3.1 모든 center가 비슷한 거리에 있다

14D 정규화 공간 [-1, 1]^14에서 uniform random으로 3000개 center를 배치하면,
임의의 query 포인트 x에서 **모든 center까지의 거리가 거의 동일**하다.

```
88,339개 실제 데이터 기준 실측값 (PHYS 수정 후):
  nearest center 거리:  mean=1.17,  p50=1.18,  p90=1.32

→ 1등 center와 3000등 center의 거리 차이가 거의 없음
```

width=0.6 기준 각 center의 phi 값:
```
phi_i = exp(-1.17² / (2 × 0.6²)) ≈ 0.20   (최근접 center)
phi_j = exp(-1.25² / (2 × 0.6²)) ≈ 0.17   (2000번째 center)

→ 가장 가까운 center와 먼 center의 phi 차이: 0.20 vs 0.17 (17% 차이뿐)
```

phi.sum() ≈ 3000 × 0.19 = 570

```
phi_norm_i ≈ 0.20 / 570 ≈ 0.00035   (각 center가 0.035%씩 기여)
```

**결과**: 3000개 center가 거의 균등하게 output에 기여 → output ≈ mean(all weights)

### 3.2 학습 후 실제 출력값 계산

14D PHYS 공간에서 실제 데이터가 차지하는 부피 비율:
```
7개 non-velocity 차원에서 평균 70% 사용
0.7^7 × 1.0^4 ≈ 0.8% (나머지 7개는 100% 사용이 아니라 다 보정해야 하는데 대략)

실제 데이터 영역에 있는 center 수:  3000 × 0.8% ≈ 24개
```

학습이 완료되어 24개 center의 weight이 5.0 → 0.2로 낮아졌다고 가정:
```
output = Σ(phi_norm_i × weight_i)
       = (24개 학습된 center × 0.00035 × 0.2)
       + (2976개 미학습 center × 0.00035 × 5.0)

       = 0.0017  +  5.208 × 0.00035 × 5.0 × 2976/1
       
       실제:
       = 24 × 0.00035 × 0.2   = 0.0017
       + 2976 × 0.00035 × 5.0 = 5.208
       (합산 정규화 후)
       ≈ 4.96
```

### 3.3 결론: in-distribution과 OOD가 모두 ≈ 5.0

```
in-distribution 출력 ≈ 4.96   → 4.96 > threshold=1.5 → truncate ❌
OOD 출력            ≈ 5.0    → 5.0  > threshold=1.5 → truncate ❌
```

**두 상태를 구별할 수 없다.** 모든 rollout이 첫 step에서 truncate → rollout_avg_horizon ≈ 0.

이 문제는 **width를 어떻게 바꿔도, PHYS를 어떻게 수정해도 해결되지 않는다.**
소프트맥스 정규화가 14D의 "거리 균질화"와 결합되어 항상 mean(weights)를 출력하기 때문이다.

---

## 4. Width 선택의 딜레마 (uniform centers 기준)

| width | phi>0.1인 데이터 비율 | 문제 |
|---|---|---|
| 0.4 | 0.4% | 대부분 데이터에서 모든 center phi≈0 → 학습 안 됨 |
| 0.6 | 83.6% | 학습은 되지만 3번 문제 (소프트맥스 균등화) 그대로 |
| 0.8 | 100% | 학습되지만 구별력 없음 |

어떤 width를 써도 근본 문제(소프트맥스 균등화)는 해결되지 않는다.

---

## 5. Data-centered Centers도 실패하는 이유

이전 분석(구 섹션 5)에서 "data-centered centers → OOD 탐지 성공"이라 주장했으나, **이는 틀렸다.**

### 왜 실패하는가

centers를 real buffer에서 3000개 샘플링하면:
- 모든 3000개 center가 실제 데이터 위치에 있음
- 학습을 통해 **3000개 전부** 낮은 uncertainty target으로 수렴
- weights → 모두 0.1~0.3 (낮은 값)

OOD 상태 x에 대해:
- 모든 center(=real data 위치)와 멀리 떨어져 있음 → phi_i ≈ 매우 작음
- 하지만 phi가 작더라도 소프트맥스는 여전히 작동:
  ```
  phi_norm_i = phi_i / Σphi_j ≈ 균등 분포 (모두 작으면 비율이 균등)
  output ≈ mean(all weights) ≈ mean(0.1 ~ 0.3) ≈ 0.2
  ```
- output ≈ 0.2 < threshold=1.5 → **OOD 통과 (탐지 실패!)**

### 두 방식 비교

| 방식 | in-distribution 출력 | OOD 출력 | 결과 |
|---|---|---|---|
| Uniform centers | ≈ 5.0 (미학습 center 지배) | ≈ 5.0 | 전부 truncate |
| Data-centered centers | ≈ 0.2 (학습된 center) | ≈ 0.2 (소프트맥스 균등화) | 전부 통과 |

**양쪽 모두 구별 불가능.** 소프트맥스 정규화가 공통 원인이다.

---

## 6. 해결: Coverage Estimator (✅ 구현 완료)

### 6.1 핵심 아이디어

"uncertainty 예측"이 아닌 **"데이터 커버리지 측정"** 으로 목적 변경.

```python
# 기존: 학습된 weight 출력 → 소프트맥스로 인해 항상 mean(weights)
output = phi_norm @ weights          # ❌ 소프트맥스 균등화 → 구별 불가

# 변경: 최대 활성화값 자체가 커버리지 지표
coverage = phi.max()                 # ✅ 정규화 없음, weight 없음
```

- in-distribution: 가까운 data center 있음 → phi.max() 높음 → 통과
- OOD: 모든 data center와 멀리 떨어짐 → phi.max() 낮음 → truncate

소프트맥스 정규화 불필요. **weight 학습도 불필요.**

### 6.2 실측 검증 (88,339 샘플, width=0.30, K=3000)

```
in-distribution (real data):
  phi.max() p5  = 0.131  ← coverage_threshold (5th percentile)
  phi.max() p50 = 0.494  ← 대부분의 학습 데이터

OOD 시나리오 (truncate 비율):
  z=0.65 (정상 범위 0.70 미만): phi.max() mean=0.0083 → 100% truncated ✓
  z=0.50 (심각한 낙하):          phi.max() mean=0.0007 → 100% truncated ✓
  angle=0.30 (정상 ±0.20 초과):  phi.max() mean=0.0107 → 100% truncated ✓
```

### 6.3 구현 흐름

```
[rollout interval마다]
  1. update_coverage_rbf(uncert_model, real_buffer, percentile=5.0)
       ├─ set_centers(obs_act)      : real buffer 샘플로 centers 교체
       └─ compute_threshold(obs_act): coverage 5th percentile 계산 → threshold
  
  2. generate_samples_rbf(... options={'coverage_threshold': threshold})
       for each rollout step:
         coverage = uncert_model(obs_act)   # phi.max() (batch,)
         is_reliable = coverage >= threshold
         truncate if not is_reliable
```

### 6.4 장점

| 구 RBF (softmax + weight 학습) | Coverage Estimator |
|---|---|
| in-dist와 OOD 모두 ≈5.0 (구별 불가) | in-dist≈0.5, OOD≈0.01 (명확 구별) |
| 소프트맥스 균등화 문제 | 정규화 없음 (max만 사용) |
| gradient 학습 필요 (수렴 불안정) | 학습 불필요 (center 위치만 중요) |
| width/PHYS 조정에도 해결 안 됨 | data-centered centers → 항상 작동 |

### 6.5 트레이드오프

- policy가 개선되어 새 영역 방문 → coverage 일시적으로 낮음 → 해당 rollout truncate
- **이는 올바른 동작**: 미방문 영역은 model이 학습 안 된 곳 → 보수적 truncation 타당
- 다음 rollout interval에 centers 재샘플링 → 새 영역 자동 포함

---

## 7. 비정상성(Non-stationarity) 문제

policy가 개선될수록 초기 탐색 시 방문하지 않았던 새로운 state 영역을 방문하게 됨.

Coverage Estimator 방식에서는:
- centers = real buffer에서 샘플링
- rollout interval(250 env steps)마다 real buffer가 갱신됨
- **rollout interval마다 centers 재샘플링** → 자동으로 새 영역 포함

기존 방식 대비 단순화:
- weight 리셋 불필요
- 학습 수렴 대기 불필요
- centers 재샘플만으로 충분

---

## 8. 현재 상태 요약 (2026-06-10)

| 항목 | 상태 |
|---|---|
| PHYS 범위 수정 | ✅ 완료 (thigh/leg clipping 제거, 실제 데이터 기반) |
| 소프트맥스 균등화 문제 | ✅ 해결 (phi.max() + data-centered centers) |
| Coverage Estimator 구현 | ✅ 완료 (`normalized_rbf.py` 전면 교체) |
| weight 학습 제거 | ✅ 완료 (`rbf_opt`, `num_rbf_epochs`, `lr_rbf` 삭제) |
| config 정리 | ✅ 완료 (`rbf_width=0.30`, `rbf_coverage_percentile=5.0`) |
| CSV 키 업데이트 | ✅ 완료 (`rbf_loss`→`coverage_threshold`, `buf_recomp_uncert_avg`→`buf_coverage_avg`) |

**모니터링 지표 해석:**
- `coverage_threshold`: 매 rollout interval에 real data 5th percentile로 계산 (≈0.13)
- `buf_coverage_avg`: real data 평균 coverage (≈0.5, 높을수록 centers 잘 커버)
- `rollout_rbf_pass_avg`: 통과 스텝의 평균 coverage (≈0.3~0.9, 높을수록 정상)
- `rollout_rbf_trunc_avg`: truncate 스텝의 평균 coverage (≈0~0.05, 낮을수록 OOD 잘 탐지)
