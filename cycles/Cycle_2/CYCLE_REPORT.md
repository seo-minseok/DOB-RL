# Cycle 2 실험 보고서

> 생성일: 2026-04-23
> 시작점: base

## 1. 변경점 요약

CartPole (discrete) → BipedalWalker-v3 (continuous) 환경 전환.
알고리즘을 DQN → TD3 (Twin Delayed DDPG)로 교체하고, DOB 차원을 2D → 7D로 확장.

## 2. 변경 상세

| 파일 | 변경 내용 |
|---|---|
| `dob_mbrl/dynamics/constants.py` | CartPole 상수 → BipedalWalker 상수. OBS_DIM=24, ACT_DIM=4, DOB_DIM=7. FPINV(7×24), F_MAT(24×7). IN_MIN/MAX 28D. |
| `dob_mbrl/dynamics/nominal.py` | `step_nominal_cartpole` → `step_nominal_bipedalwalker`. 운동학 모델: 각도 = 이전각도 + Ts × 각속도, 가속도=0 가정. |
| `dob_mbrl/dynamics/dob.py` | `action_force: float` → `action: np.ndarray (4,)`. dhat/uncertainty 7D. |
| `dob_mbrl/dynamics/__init__.py` | 신규 심볼 export. ACT_ELEMENTS/FORCE_MAG 제거. |
| `dob_mbrl/envs/bipedalwalker_utils.py` | **신규**. `make_bipedalwalker_env`, `reset_env`, `step_env`. `reward_is_done_function` (model rollout 근사). |
| `dob_mbrl/envs/__init__.py` | cartpole_utils → bipedalwalker_utils. |
| `dob_mbrl/models/actor_network.py` | **신규**. TD3 Actor: obs(24) → tanh(FC256×2 → FC4). |
| `dob_mbrl/models/q_network.py` | Q(s) discrete → Q(s,a) continuous critic. Input 28D, output scalar. |
| `dob_mbrl/models/residual_dx_net.py` | input 5D→28D, output 2D→7D (DOB_DIM). hidden 32→64. |
| `dob_mbrl/models/normalized_rbf.py` | Centers 28D, weights (7×K). CartPole의 half/half act split 제거 → uniform 샘플링. |
| `dob_mbrl/models/__init__.py` | ActorNetwork 추가. |
| `dob_mbrl/utils/buffer.py` | num_obs=24, num_act=4 기본값. dhat/uncertainty (N, DOB_DIM=7). |
| `dob_mbrl/training/config.py` | epsilon 계열 제거. TD3 파라미터 추가: lr_actor, policy_delay, expl_noise, policy_noise, noise_clip. num_episodes=2000, max_steps_per_ep=1600, warm_start_samples=10000. |
| `dob_mbrl/training/model_learning.py` | dhat/uncertainty shape 변경에 따라 주석 수정. 로직 동일. |
| `dob_mbrl/training/rollout.py` | `q_network` 파라미터 → `actor`. 행동 선택: argmax → actor 출력 + Gaussian 탐색 노이즈. |
| `dob_mbrl/training/trainer.py` | DQN → TD3 전면 재작성. actor/2×critic, target network, policy delay, soft update 분리. env reward 직접 사용. |
| `run_multi_seed.py` | target_score 480 → 300 (BipedalWalker 해결 기준). |
| `scripts/plot_results.py` | Target line 480 → 300. |

## 3. 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---|---|---|
| `num_episodes` | 2000 | BipedalWalker는 긴 학습 필요 |
| `max_steps_per_ep` | 1600 | gymnasium 기본값 |
| `warm_start_samples` | 10000 | 무작위 탐색 구간 |
| `lr_actor` / `lr_critic` | 3e-4 | TD3 표준 |
| `tau` | 0.005 | soft update |
| `policy_delay` | 2 | 액터 1/2 빈도 업데이트 |
| `expl_noise` | 0.1 | 환경 탐색 노이즈 std |
| `policy_noise` | 0.2 | 타깃 정책 노이즈 std |
| `noise_clip` | 0.5 | 타깃 정책 노이즈 클리핑 |
| `update_interval` | 10 | 10 env steps마다 업데이트 |
| `num_gradient_steps` | 1 | UTD ≈ 1 |
| `buffer_size` | 1e6 | 연속 제어 표준 |
| `real_ratio` | 0.2 | 20% real / 80% model |
| `DOB_DIM` | 7 | hull_angvel, vel_x, vel_y, hip/knee speeds |
| `num_rbf_centers` | 600 | 기존 유지 |

## 4. 학습 결과
<!-- 에이전트 자동 생성: 정량 지표 + figure 파일 경로 목록 -->
*(학습 실행 후 채워질 예정)*

## 5. 관찰 및 교훈
<!-- ★ 사람이 직접 작성 — 에이전트가 빈 섹션으로 예약 -->

## 6. 메모리 업데이트 제안 (선택)
<!-- 에이전트가 필요 시 자동 생성 -->
