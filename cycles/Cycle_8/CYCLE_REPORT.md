# Cycle 8 실험 보고서

> 생성일: 2026-05-26
> 시작점: Cycle_7

## 1. 변경점 요약

- **환경 교체**: MountainCarContinuous-v0 → Gymnasium Hopper-v5 (MuJoCo)
- **관측/행동 차원 변경**: obs 2D → 11D, act 1D → 3D
- **DOB 설계 변경**: DOB_DIM 1 → 11 (전체 state 추적, FPINV = F_MAT = Identity)
- **Nominal 모델**: kinematic Euler integration — obs[0:5](위치) += obs[6:11](속도) × dt, 속도 고정
- **알고리즘은 Cycle 7 유지**: DDPG + DOB + MBRL (real_ratio, uncertainty sampling)

## 2. 변경 상세

| 파일 | 변경 내용 |
|---|---|
| `dob_mbrl/envs/hopper_utils.py` | 신규 생성 — Hopper-v5 env, synthetic reward, done 조건 |
| `dob_mbrl/envs/__init__.py` | hopper_utils로 임포트 교체 |
| `dob_mbrl/dynamics/constants.py` | obs_dim=11, act_dim=3, FPINV=F_MAT=I_11, healthy bounds |
| `dob_mbrl/dynamics/nominal.py` | kinematic nominal: obs[0:5] += obs[6:11] * dt, 속도 고정 |
| `dob_mbrl/dynamics/dob.py` | DOB_DIM=11, step_nominal_hopper로 교체 |
| `dob_mbrl/dynamics/__init__.py` | Hopper 함수명으로 export 갱신 |
| `dob_mbrl/training/config.py` | max_steps=1000, expl_noise=0.3, horizon=5, uncert_thresh=0.1 |
| `dob_mbrl/training/trainer.py` | NUM_OBS=11, NUM_ACT=3, ResidualDxNet hidden=64, TARGET_SCORE=500 |
| `dob_mbrl/training/rollout.py` | hopper_utils reward로 임포트 교체 |

## 3. 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---|---|---|
| env | Hopper-v5 | MuJoCo |
| num_episodes | 1000 | |
| max_steps_per_ep | 1000 | |
| warm_start_samples | 5000 | |
| expl_noise | 0.3 | MountainCar(0.5)보다 낮춤 |
| DOB_DIM | 11 | 전체 state 추적 |
| FPINV / F_MAT | I_11 | identity |
| max_horizon_length | 5 | MountainCar(10)보다 줄임 |
| uncertainty_threshold | 0.1 | |
| TARGET_SCORE | 500.0 | Hopper-v5 체크포인트 저장 기준 |
| res_net hidden | 64 | MountainCar(32)보다 넓힘 |

## 4. 학습 결과

<!-- ★ 사람이 직접 작성 -->

## 5. 관찰 및 교훈

<!-- ★ 사람이 직접 작성 -->

## 6. 메모리 업데이트 제안 (선택)

<!-- ★ 사람이 직접 작성 -->
