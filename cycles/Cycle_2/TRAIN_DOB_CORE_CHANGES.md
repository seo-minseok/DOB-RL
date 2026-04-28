# Cycle 1 → Cycle 2 코드 변경 상세 분석

> **대상**: `train_DOB_core` 로직 전체 (monolithic → modular 구조)  
> **핵심 변화**: CartPole-v1 (discrete, 4D obs) → BipedalWalker-v3 (continuous, 24D obs)  
> **알고리즘 변화**: DQN → TD3 (Twin Delayed DDPG)

---

## 1. 환경 변화

| 항목 | Cycle 1 (CartPole-v1) | Cycle 2 (BipedalWalker-v3) |
|---|---|---|
| 관측 차원 (`OBS_DIM`) | 4 | 24 |
| 행동 차원 (`ACT_DIM`) | 1 (이산 → force scalar로 변환) | 4 (연속, 각 모터) |
| 행동 공간 | Discrete(2): index 0 또는 1 | Continuous: 4D ∈ [-1, 1] |
| DOB 추적 차원 (`DOB_DIM`) | 2 (cart velocity, pole angular velocity) | 7 (속도 성분 7개) |
| 에피소드 최대 스텝 | 500 | 1600 |
| 보상 함수 | 커스텀 shaped reward (각도/위치 기반) | env 보상 직접 사용 |

---

## 2. 상수 / 차원 변화 (`constants.py`)

### Cycle 1 — CartPole

```python
OBS_DIM = 4  # [pos, vel, theta, thetadot]
ACT_DIM = 1  # force scalar

# FPINV: (2, 4) — velocity 2개 추출
FPINV = [[0, 1, 0, 0],
         [0, 0, 0, 1]]

# F_MAT: (4, 2) — residual을 obs 공간으로 역투영
F_MAT = [[0, 0], [1, 0], [0, 0], [0, 1]]

# 입력 범위: obs(4) + act(1) = 5D
IN_MIN = [*OBS_MIN, -FORCE_MAG]   # (5,)
IN_MAX = [*OBS_MAX,  FORCE_MAG]   # (5,)
```

### Cycle 2 — BipedalWalker

```python
OBS_DIM = 24
ACT_DIM = 4
DOB_DIM = 7  # velocity 인덱스: [1, 2, 3, 5, 7, 10, 12]

# FPINV: (7, 24) — obs에서 7개 velocity 성분 추출 (단위 행렬 행 선택)
VELOCITY_INDICES = [1, 2, 3, 5, 7, 10, 12]
FPINV = zeros((7, 24))
FPINV[i, VELOCITY_INDICES[i]] = 1.0  # (각 행에 1이 하나)

# F_MAT: (24, 7) — FPINV.T
F_MAT = FPINV.T

# 입력 범위: obs(24) + act(4) = 28D
IN_MIN = [*OBS_MIN, *ACT_MIN]   # (28,)
IN_MAX = [*OBS_MAX, *ACT_MAX]   # (28,)
```

**핵심 변화 이유**: BipedalWalker의 obs 24개 중 실제로 DOB가 추정해야 할 성분은 가속도의 영향을 받는 velocity 7개뿐이다. CartPole과 동일한 원리로 "velocity 인덱스만 FPINV로 추출"하되, 차원이 2→7로 확장된 것.

---

## 3. 강화학습 알고리즘 변화: DQN → TD3

### 3-1. Cycle 1: DQN (Discrete)

```
QNetwork(obs=4) → Q-values (batch, 2)
                         ↑
              2개 이산 행동에 대한 Q값

Action selection: argmax(Q(s))
Critic update  : act_mask = (ACT_ELEMENTS == action_force)
                 q_pred = (Q(s) * act_mask).sum(dim=1)  ← 선택된 행동의 Q값만 추출
```

- 출력이 행동별 Q값 벡터 → argmax로 행동 선택
- 타깃: `max_q_next = target_network(next_obs).max(dim=1).values`

### 3-2. Cycle 2: TD3 (Continuous)

```
ActorNetwork(obs=24) → action (batch, 4), tanh ∈ [-1, 1]

QNetwork(obs=24, act=4) → Q-value (batch,) scalar
                ↑
        obs와 action을 함께 입력

Critic 1, Critic 2 (Twin Critics)
Target Actor + Target Critic 1 + Target Critic 2

Critic update: Double-Q → min(Q1_next, Q2_next)으로 과대 추정 억제
Actor update : -Critic1(s, actor(s)).mean()  (every policy_delay steps)
```

**TD3의 3가지 핵심 기법이 추가됨:**

| 기법 | 목적 | 구현 위치 |
|---|---|---|
| Twin Critics (Clipped Double Q) | Q값 과대 추정 억제 | `target_q = min(Q1_next, Q2_next)` |
| Delayed Policy Update | actor 업데이트 안정화 | `if total_grad_steps % policy_delay == 0` |
| Target Policy Smoothing | 날카로운 Q 함수 방지 | `noise_t = clamp(randn * policy_noise, ±noise_clip)` |

---

## 4. 네트워크 아키텍처 변화

### 4-1. QNetwork

| 항목 | Cycle 1 | Cycle 2 |
|---|---|---|
| 입력 | obs (4D) | obs (24D) + act (4D) = 28D |
| 출력 | Q-values (2D, 행동별) | scalar Q-value (1D) |
| 은닉층 크기 | 128 | 256 |
| forward signature | `forward(obs)` | `forward(obs, act)` |
| obs 정규화 | O | O |
| act 정규화 | — | 없음 ([-1,1] 범위라 불필요) |

**Cycle 1:**
```python
class QNetwork(nn.Module):
    def __init__(self, num_observations=4, num_actions=2):
        self.fc1 = nn.Linear(num_observations, 128)   # 4 → 128
        self.fc3 = nn.Linear(128, num_actions)         # 128 → 2
    def forward(self, obs):
        obs_norm = normalize(obs)
        return fc3(relu(fc2(relu(fc1(obs_norm)))))    # (batch, 2)
```

**Cycle 2:**
```python
class QNetwork(nn.Module):
    def __init__(self, num_obs=24, num_act=4):
        self.fc1 = nn.Linear(num_obs + num_act, 256)  # 28 → 256
        self.fc3 = nn.Linear(256, 1)                   # 256 → 1
    def forward(self, obs, act):
        obs_norm = normalize(obs)
        x = cat([obs_norm, act], dim=-1)               # (batch, 28)
        return fc3(relu(fc2(relu(fc1(x))))).squeeze(-1) # (batch,)
```

### 4-2. ActorNetwork (Cycle 2 신규 추가)

```python
class ActorNetwork(nn.Module):
    def __init__(self, num_obs=24, num_act=4):
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_act)

    def forward(self, obs):
        obs_norm = normalize(obs)
        x = relu(fc1(obs_norm))
        x = relu(fc2(x))
        return tanh(fc3(x))   # (batch, 4) ∈ [-1, 1]
```

- Cycle 1에는 Actor가 없음 (DQN은 Q값만으로 action 결정)
- tanh 출력으로 BipedalWalker의 `[-1, 1]` 행동 범위 만족

### 4-3. ResidualDxNet

| 항목 | Cycle 1 | Cycle 2 |
|---|---|---|
| 입력 | obs(4) + act(1) = 5D | obs(24) + act(4) = 28D |
| 출력 | 2D (vel, thetadot residual) | 7D (velocity 7성분 residual) |
| 은닉층 크기 | 32 | 64 |

### 4-4. NormalizedRBFModel

| 항목 | Cycle 1 | Cycle 2 |
|---|---|---|
| 입력 차원 | 5D | 28D |
| centers 행렬 크기 | (5, K) | (28, K) |
| weights 크기 | (2, K) | (7, K) |
| 출력 | (batch, 2) | (batch, 7) |
| centers 초기화 | 특수 처리 (state 4D + act 2-극점) | uniform random in [phys_min, phys_max] |

Cycle 1에서는 action centers를 `-10 / +10` 두 점으로 고정했었다. Cycle 2에서는 행동 공간이 4D 연속이므로 단순 uniform 샘플링으로 통일.

---

## 5. Nominal Dynamics 변화

### Cycle 1: `step_nominal_cartpole`

```
물리 모델: pos, vel, theta, thetadot
가속도(posdd, thdd) = 0 가정

xdot = [vel, 0, thetadot, 0]
x_next = x + Ts * xdot
```

### Cycle 2: `step_nominal_bipedalwalker`

```
운동학 모델: 24D observation
각속도 → 각도 적분 (5쌍)

hull_angle    += Ts * hull_angvel
hip1_angle    += Ts * hip1_speed
knee1_angle   += Ts * knee1_speed
hip2_angle    += Ts * hip2_speed
knee2_angle   += Ts * knee2_speed

velocity, contact, lidar는 변경 없이 그대로 복사
```

**Cycle 2에서 action을 사용하지 않음**: BipedalWalker의 명목 모델은 순수 운동학(kinematics)이라 action(모터 토크)의 영향을 무시한다. DOB와 residual net이 실제 dynamics(토크 효과)를 보정.

---

## 6. 탐색(Exploration) 전략 변화

### Cycle 1: ε-greedy (이산 행동)

```python
# warm_start_samples = 200 이후 epsilon 감쇠
if rand() < epsilon:
    action_idx = randint(0, 2)          # 무작위 이산 행동
else:
    action_idx = argmax(Q(obs))         # greedy

action_force = ACT_ELEMENTS[action_idx]  # [-10, 10] 중 하나
gym_action   = action_idx                # env에 0 또는 1 전달
```

### Cycle 2: Gaussian Exploration (연속 행동)

```python
# warm_start_samples = 10000 이전: 완전 무작위
if total_step_ct <= warm_start_samples:
    action = uniform(-1.0, 1.0, size=4)

# 이후: actor + Gaussian noise
else:
    action = actor(obs).numpy()
    noise  = normal(0.0, expl_noise=0.1, size=4)
    action = clip(action + noise, -1.0, 1.0)
```

| 항목 | Cycle 1 | Cycle 2 |
|---|---|---|
| 탐색 방식 | ε-greedy | Gaussian noise |
| warm-start 길이 | 200 steps | 10,000 steps |
| 탐색 파라미터 | epsilon: 1.0 → 0.01 (decay) | expl_noise: 0.1 (고정) |

---

## 7. Rollout 생성 변화 (`generate_samples_dob`)

### Cycle 1: Q-network argmax + epsilon-greedy 대체

```python
# argmax로 행동 선택
action_idx = q_network(valid_obs).argmax(dim=1).numpy()
valid_act  = ACT_ELEMENTS[action_idx].reshape(-1, 1)  # (n_alive, 1)

# epsilon-greedy 대체
rand_act     = ACT_ELEMENTS[randint(0, 2, n_alive)].reshape(-1, 1)
replace_mask = rand() < eps_model
valid_act[replace_mask] = rand_act[replace_mask]
```

### Cycle 2: Actor + Gaussian noise

```python
# actor 출력
valid_act_t = actor(valid_obs)    # (n_alive, 4), tanh ∈ [-1, 1]
valid_act   = valid_act_t.numpy()

# Gaussian noise 추가 후 클리핑
noise     = normal(0.0, noise_std, size=valid_act.shape)
valid_act = clip(valid_act + noise, -1.0, 1.0)
```

---

## 8. Critic 업데이트 로직 변화

### Cycle 1: DQN 스타일 (act_mask)

```python
# act_mask: 선택된 행동에 해당하는 Q값만 추출
act_mask = (ACT_ELEMENTS == act_bt).float()   # (batch, 2)
q_pred   = (Q(obs_bt) * act_mask).sum(dim=1)   # (batch,)

# 타깃: 다음 obs의 max Q
max_next_q = target_network(nxt_bt).max(dim=1).values
target_q   = rew + γ * max_next_q
```

### Cycle 2: TD3 스타일 (Double Q + Target Policy Noise)

```python
# Target Policy Noise
noise_t = clamp(randn_like(act_bt) * policy_noise, -noise_clip, noise_clip)
tgt_act = clamp(target_actor(nxt_bt) + noise_t, -1.0, 1.0)

# Double Q: min(Q1, Q2)으로 과대추정 억제
q1_next  = target_critic1(nxt_bt, tgt_act)
q2_next  = target_critic2(nxt_bt, tgt_act)
target_q = rew + γ * min(q1_next, q2_next)

# Critic 1, 2 각각 업데이트
loss_c1 = mse(critic1(obs_bt, act_bt), target_q)
loss_c2 = mse(critic2(obs_bt, act_bt), target_q)
```

---

## 9. Actor 업데이트 (Cycle 2 신규)

```python
# policy_delay(=2) 스텝마다 actor 업데이트
if total_grad_steps % policy_delay == 0:
    actor_loss = -critic1(obs_bt, actor(obs_bt)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    clip_grad_norm_(actor.parameters(), 1.0)
    actor_opt.step()

    # Soft update: actor + critic1 + critic2 모두
    soft_update(actor,   target_actor,   tau)
    soft_update(critic1, target_critic1, tau)
    soft_update(critic2, target_critic2, tau)
```

Cycle 1에서는 soft update가 매 gradient step마다 Q-network에만 적용됐다.  
Cycle 2에서는 3개 target network 모두 actor delay 주기와 동기화.

---

## 10. 체크포인트 변화

### Cycle 1

```python
# 저장 조건: 10-ep avg >= 480 AND 새 best
torch.save({
    'q_network'   : q_network.state_dict(),
    'res_net'     : res_net.state_dict(),
    'uncert_model': uncert_model.state_dict(),
    'epsilon'     : epsilon,
    ...
})
```

### Cycle 2

```python
# 저장 조건: 10-ep avg 새 best (점수 하한 없음)
torch.save({
    'actor'            : actor.state_dict(),
    'critic1'          : critic1.state_dict(),
    'critic2'          : critic2.state_dict(),
    'res_net'          : res_net.state_dict(),
    'uncert_model'     : uncert_model.state_dict(),
    'total_grad_steps' : total_grad_steps,
    ...
})
```

- `q_network` → `actor + critic1 + critic2` (TD3 구조 반영)
- epsilon 제거 (탐색이 고정 noise로 바뀌었으므로)
- BipedalWalker는 점수 범위가 불규칙하므로 절대 임계값 조건 제거

---

## 11. 주요 하이퍼파라미터 비교

| 파라미터 | Cycle 1 | Cycle 2 |
|---|---|---|
| `buffer_size` | 1e5 | **1e6** (10배) |
| `warm_start_samples` | 200 | **10,000** (50배) |
| `max_steps_per_ep` | 500 | **1,600** |
| `lr_critic` | 1e-3 | **3e-4** |
| `lr_actor` | — | 3e-4 (신규) |
| `num_gradient_steps` | 2 (UTD=20) | **1** (UTD=10) |
| `policy_delay` | — | **2** (신규) |
| `expl_noise` | — | **0.1** (신규) |
| `policy_noise` | — | **0.2** (신규) |
| `noise_clip` | — | **0.5** (신규) |
| `epsilon` | 1.0 → 0.01 | 제거 |
| ResidualDxNet hidden | 32 | **64** |
| 네트워크 hidden | 128 | **256** |

---

## 12. 코드 구조 변화 (monolithic → modular)

Cycle 1은 `base/original/train_DOB_core.py` 단일 파일에 모든 클래스/함수가 집중되어 있었다.  
Cycle 2에서는 `dob_mbrl` 패키지로 분리:

```
cycles/Cycle_2/dob_mbrl/
├── dynamics/
│   ├── constants.py       # BipedalWalker 차원·상수 (OBS_DIM=24, ACT_DIM=4, DOB_DIM=7)
│   ├── nominal.py         # step_nominal_bipedalwalker
│   └── dob.py             # predict_next_obs_dob, compute_dob_update
├── models/
│   ├── actor_network.py   # ActorNetwork (TD3 신규)
│   ├── q_network.py       # QNetwork (Q(s,a) scalar)
│   ├── residual_dx_net.py # ResidualDxNet (28→7)
│   └── normalized_rbf.py  # NormalizedRBFModel (28D input)
├── envs/
│   └── bipedalwalker_utils.py  # make_env, reset_env, step_env, reward_is_done
├── training/
│   ├── config.py          # DOBMBRLConfig (TD3 파라미터 포함)
│   ├── model_learning.py  # train_residual_dx_model_dob, train_uncertainty_rbf
│   ├── rollout.py         # generate_samples_dob (actor 기반), sample_mixed_minibatch
│   └── trainer.py         # train_DOB_core 메인 루프
└── utils/
    └── buffer.py          # ReplayBufferDOB
```

---

## 요약

| 변화 범주 | Cycle 1 | Cycle 2 |
|---|---|---|
| 환경 | CartPole-v1 (4 obs, discrete 2) | BipedalWalker-v3 (24 obs, continuous 4) |
| 알고리즘 | DQN | TD3 |
| Policy Network | 없음 (Q argmax) | ActorNetwork (tanh, 4D output) |
| Critic 구조 | Q(s) → (batch, 2) | Q(s,a) → (batch,) scalar, Twin |
| DOB 차원 | 2D (velocity 2) | 7D (velocity 7) |
| FPINV | (2, 4) | (7, 24) |
| 탐색 | ε-greedy | Gaussian noise |
| 보상 | 커스텀 shaped | env 보상 직접 사용 |
| 네트워크 크기 | hidden=128, res=32 | hidden=256, res=64 |
| Warm-start | 200 steps | 10,000 steps |
