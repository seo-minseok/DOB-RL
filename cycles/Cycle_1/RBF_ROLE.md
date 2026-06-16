# RBF Uncertainty Model — 역할 및 구현 분석

## 한 줄 요약

> **res_net이 이 (obs, act)에서 얼마나 틀리는지를 학습해,  
> rollout 중 모델 신뢰도가 낮은 구간을 조기 종료(truncate)한다.**

---

## 1. 학습 타깃

`train_uncertainty_rbf` (`model_learning.py:81-130`) 에서 매 rollout interval마다 학습.

```
dx_real  = next_obs - obs               # 실제 dynamics 변화량
e        = dx_real - dx_nom             # nominal이 설명 못한 부분
fpinv_e  = FPINV @ e                    # DOB 공간으로 변환 (FPINV = I)
dx_res   = res_net(obs, act)            # residual network 예측값

fresh_uncert = fpinv_e - dx_res         # res_net의 실제 예측 오차  ← RBF 타깃
```

RBF는 `(obs, act) → ||fresh_uncert||` 함수를 근사한다.  
즉 "이 상태/행동에서 res_net이 실제로 얼마나 틀렸는가"를 공간상의 함수로 기억한다.

---

## 2. 모델 구조 (NormalizedRBFModel)

`models/normalized_rbf.py`

| 구성 요소 | 설명 |
|---|---|
| 입력 | `[obs(11), act(3)]` = 14차원 |
| 출력 | `(batch, DOB_DIM=11)` — 각 state 차원별 uncertainty |
| Centers | 초기화 시 14D 공간에 3000개 균일 샘플링, **고정 (학습 안 됨)** |
| Weights | `(11, 3000)`, 초기값 `rbf_initial_value = 5.0`, **이것만 학습** |

**Forward 계산:**

```
phi[i,k] = exp(-||x_norm[i] - center[k]||² / (2σ²))   # RBF 활성화
phi_norm  = phi / (sum(phi) + ε)                         # 정규화
output    = phi_norm @ weights.T                          # (batch, 11)
```

---

## 3. OOD(분포 외) 입력에서의 동작 — 핵심 설계

입력이 학습 데이터 분포에서 멀면 → 모든 center까지 거리가 멈 → 모든 `phi[k] ≈ 0`

```
phi_norm ≈ uniform(1/K)   (모든 basis가 동등하게 작음)
output   ≈ mean(weights)  ≈ initial_value = 5.0
```

**초기값 5.0이 "모름"의 표현이다.**  
학습이 진행되면서 방문한 구간의 weights는 실제 오차값으로 업데이트되고,  
한 번도 방문 안 한 구간은 5.0을 유지 → rollout이 낯선 구간에 가면 자동으로 차단.

---

## 4. Rollout에서의 사용

`rollout.py:72-93` — 매 rollout step마다:

```python
pred_uncert = uncert_model(obs, act)          # RBF 예측 (batch, 11)
uncert_mag  = ||pred_uncert||₂               # L2 norm → scalar per trajectory

is_reliable = uncert_mag <= threshold         # threshold = 0.1 (현재 설정)

# 불신뢰 trajectory는 alive_mask에서 제거 → 이후 step에서 제외
alive_mask[~is_reliable] = False
```

통계 로깅:
- `rollout_uncert_avg` — 전 step의 평균 uncertainty
- `rollout_pass_rate` — threshold 통과 비율
- `rollout_avg_horizon` — trajectory당 평균 생존 step 수

---

## 5. 현재 threshold 설정의 문제

`config.py`: `uncertainty_threshold = 0.1` (고정값)

이 값이 실제 uncertainty 분포에서 **몇 퍼센타일에 해당하는지 알 수 없다.**

RBF 학습 타깃(`fresh_uncert`)의 실제 분포는 `real_buffer.uncertainty`에 축적되어 있는데,  
이 분포와 threshold의 관계를 확인하지 않고 있음.

### 문제 시나리오

```
실제 분포: p50=0.05, p75=0.12, p90=0.25, p95=0.40
threshold = 0.1  →  real transition의 ~30%도 통과 못하는 수준
```

→ rollout이 거의 모든 step에서 조기 종료 → `rollout_avg_horizon` 극히 낮음  
→ model buffer에 쌓이는 데이터가 없거나 극히 짧음  
→ MBPO의 model-based 이점이 사라짐

---

## 6. 제안: Percentile 기반 calibration

### 추가할 로깅

rollout interval마다 (`trainer.py:322-325` 블록):

```python
_umags = np.linalg.norm(real_buffer.uncertainty[:valid_len], axis=1)
ep_uncert_p50  = np.percentile(_umags, 50)
ep_uncert_p75  = np.percentile(_umags, 75)
ep_uncert_p90  = np.percentile(_umags, 90)
ep_uncert_p95  = np.percentile(_umags, 95)
ep_uncert_std  = float(_umags.std())
```

### threshold 해석표

| threshold 설정값 | 의미 |
|---|---|
| `p50` | real 전환의 50%도 통과 못하는 수준 (매우 빡빡) |
| `p75` | real 전환의 75%가 통과하는 수준 |
| `p90` | real 전환의 90%가 통과 **(권장 시작점)** |
| `p95` | real 전환의 95%가 통과 (느슨) |

### threshold 조정 기준

```
rollout_pass_rate < 0.6  →  threshold 너무 낮음, 올려야 함
rollout_pass_rate > 0.95 →  threshold 너무 높음, truncation 의미 없음
rollout_avg_horizon < 3  →  trajectory가 거의 즉시 종료됨, threshold 올려야 함
```

---

## 7. 주의: res_net 학습에 따른 분포 이동

학습 초반 → res_net 예측 나쁨 → `fresh_uncert` 값 큼 → 분포 전체가 우측  
학습 후반 → res_net 예측 좋아짐 → `fresh_uncert` 값 작아짐 → 분포 좌측으로 이동

**고정 threshold는 학습 초반엔 너무 빡빡하고, 후반엔 느슨해진다.**

percentile 로깅이 있으면 이 이동을 추적할 수 있다.

---

## 요약 흐름도

```
[Real env interaction]
(obs, act, next_obs)
        ↓
fresh_uncert = FPINV@(dx_real - dx_nom) - dx_res
        ↓
[RBF 학습]  (obs, act) → predict ||fresh_uncert||
        ↓
[Rollout]
(obs_rollout, act_rollout) → RBF → uncert_mag
        ↓
uncert_mag > threshold? → truncate : continue
```
