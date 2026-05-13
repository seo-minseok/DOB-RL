# Cycle 4 Code Fix Recommendations

## Scope

이 문서는 Cycle 4의 `DOB-MBRL` 실험, 특히 `real_ratio=0.5`, `uncertainty_threshold=0.3` 설정에서 나타난 시드별 학습 편차를 줄이기 위한 코드 레벨 수정안을 정리한 문서다.

분석 대상은 다음 경로를 기준으로 했다.

- `cycles/Cycle_4/dob_mbrl/training/trainer.py`
- `cycles/Cycle_4/dob_mbrl/training/rollout.py`
- `cycles/Cycle_4/dob_mbrl/training/model_learning.py`
- `cycles/Cycle_4/dob_mbrl/utils/buffer.py`
- `cycles/Cycle_4/dob_mbrl/envs/bipedalwalker_utils.py`
- `cycles/Cycle_4/results/_real_ratio=0.5_uncert_thresh=0.3`
- `cycles/Cycle_4/results/real_ratio=0.5_uncert_thresh=0.3`

이 문서의 목적은 "무엇이 문제인가"를 다시 설명하는 것이 아니라, 실제로 어떤 순서로 어떤 코드를 바꾸면 되는지를 구현 관점에서 정리하는 것이다.

## Summary

우선순위가 높은 문제는 아래 4가지다.

1. `model_buffer`가 오래된 synthetic transition을 계속 누적한다.
2. rollout에 들어가는 synthetic reward/done이 실제 환경과 지나치게 다르다.
3. uncertainty 학습 타깃이 stale하며 현재 `res_net` 상태를 반영하지 못한다.
4. mixed batch 사용 조건이 너무 느슨해서, 품질이 낮은 model sample도 critic 학습에 과감하게 들어간다.

가장 짧은 안정화 경로는 아래 순서를 추천한다.

1. `model_buffer refresh`
2. `stale uncertainty 제거`
3. `conditional model mixing`
4. `synthetic reward 개선`

## Recommended Order

### Phase 1: 즉시 반영 권장

- `ReplayBufferDOB.clear()` 추가
- rollout 직전 `model_buffer.clear()` 호출
- `sample_mixed_minibatch()`에서 model sample 최소 수량 검사 추가
- `train_uncertainty_rbf()`와 `evaluate_rbf_calibration()`가 현재 `res_net` 기준 uncertainty를 재계산하도록 변경

## 3. Improve Synthetic Reward and Done Logic

### Current Problem

현재 `reward_is_done_function()`은 `next_obs`만 받아서 아래 정보만 사용한다.

- `vel_x`
- `hull_angle`

즉 실제 BipedalWalker 보상의 중요한 요소가 빠져 있다.

- torque penalty
- 자세 shaping
- 착지/접지 관련 간접 효과
- progress 외의 안정성 요소

이 때문에 rollout transition이 state transition 측면에서는 그럴듯해도 reward target은 critic 학습에 잘못된 방향성을 줄 수 있다.

### Current Code Point

- `cycles/Cycle_4/dob_mbrl/envs/bipedalwalker_utils.py`

### Recommended Change

최소 수정과 중간 수정, 강한 수정의 3단계가 있다.

### Option A: 최소 수정

함수 시그니처를 바꿔서 `obs`, `act`, `next_obs`를 모두 사용한다.

```python
def reward_is_done_function(obs, act, next_obs):
    ...
```

그리고 아래 요소를 추가한다.

- 전진 shaping 근사
- hull angle penalty
- action magnitude penalty
- done condition 강화

예시:

```python
progress = 0.72 * next_obs[:, 2]
posture_penalty = 0.10 * np.abs(next_obs[:, 0])
torque_penalty = 0.03 * np.square(act).sum(axis=1)
reward = progress - posture_penalty - torque_penalty
done = np.abs(next_obs[:, 0]) > 1.4
reward = np.where(done, -100.0, reward)
```

이건 여전히 근사지만, 현재보다 critic target 왜곡이 줄어든다.

### Option B: reward model 도입

real buffer의 `(obs, act, next_obs) -> env_reward`를 학습하는 작은 network를 추가한다.

권장 이유:

- synthetic transition과 reward를 같은 distribution 위에서 다룰 수 있다.
- hand-crafted reward approximation보다 실제 env reward에 가깝다.

추천 구조:

- 입력: `obs + act + next_obs`
- 출력: scalar reward
- optimizer: `Adam`
- loss: `MSE`

### Option C: synthetic reward를 쓰지 않고 value expansion만 사용

이건 구조 변화가 커서 당장 1차 패치로는 추천하지 않지만, 장기적으로는 고려할 수 있다.

### Expected Effect

- critic target의 systematic bias 감소
- synthetic rollout이 "통과는 많이 되지만 reward가 틀린" 상태 완화

## 5. Revisit the `h == 0` Always-Pass Rule

### Current Problem

현재 rollout 첫 step에서는 uncertainty가 아무리 커도 무조건 통과시킨다.

```python
if h == 0:
    is_reliable[:] = True
```

이 로직은 "초기 state는 real buffer에서 왔으니 1-step 정도는 괜찮다"는 가정인데, 실제로는 첫 action 자체가 현재 actor와 exploration noise에 의해 생성되므로 OOD action이 될 수 있다.

### Recommended Change

아래 둘 중 하나를 추천한다.

### Option A: 완전 제거

가장 단순하다.

```python
if h == 0:
    pass
```

즉 모든 step에 동일한 threshold를 적용한다.

### Option B: 완화형 예외

`h == 0`에서도 threshold를 적용하되, 조금 더 느슨한 threshold를 사용한다.

예시:

```python
step_threshold = uncert_threshold * 1.2 if h == 0 else uncert_threshold
is_reliable = uncert_mag_arr < step_threshold
```

### Recommendation

1차 패치에서는 Option B가 더 안전하다.

### Expected Effect

- 첫 step synthetic sample이 대량으로 무비판적으로 주입되는 문제 완화
- 특히 실패 시드에서 model buffer 오염 속도 감소

## 6. Change Residual Model Target

### Current Problem

현재 `res_net`은 `dhat`을 타깃으로 학습한다.

하지만 `dhat`은 아래처럼 정의된다.

```python
dhat = dob_w * dx_res + (1 - dob_w) * (FPINV @ e)
```

즉 현재 `dx_res` 자신이 이미 섞인 값이다. 이 구조는 target이 순수 ground truth residual이 아니라 self-referential한 moving target이 되는 문제가 있다.

### Recommended Change

`res_net`은 더 직접적인 residual target을 학습하도록 바꾼다.

### Candidate Target

```python
target_residual = FPINV @ (dx_real - dx_nom)
```

batch 기준으로는 아래와 같은 형태가 된다.

```python
dx_real = next_obs - obs
e = dx_real - dx_nom
target = e[:, VELOCITY_INDICES]
```

또는 `target = e @ FPINV.T` 방식으로 구현해도 된다.

### Why This Is Better

- target이 현재 model output에 덜 의존한다.
- 학습 signal이 더 직접적이다.
- uncertainty 정의도 더 일관되게 정리할 수 있다.

### Migration Plan

1. `train_residual_dx_model_dob()`에서 `real_buffer.dhat` 대신 `obs`, `next_obs`, `dx_nom`을 사용
2. helper로 target batch 계산
3. uncertainty-weighted sampling은 유지하되, 타깃만 변경

### Note

이 수정은 효과가 클 수 있지만 동작 의미가 약간 바뀌므로, 1차 안정화 후 별도 ablation으로 비교하는 것이 좋다.

## 7. Seed the Environment Explicitly

### Current Problem

지금은 `np.random.seed(run_idx)`와 `torch.manual_seed(run_idx)`만 고정한다. 하지만 환경 seed는 명시적으로 고정하지 않는다.

### Recommended Change

`trainer.py`에서 env 생성 직후 아래를 적용한다.

```python
env = make_bipedalwalker_env()
env.reset(seed=run_idx)
if hasattr(env.action_space, 'seed'):
    env.action_space.seed(run_idx)
if hasattr(env.observation_space, 'seed'):
    env.observation_space.seed(run_idx)
```

그리고 episode reset helper도 seed를 받을 수 있게 확장할 수 있다.

### Expected Effect

- 시드별 차이를 "알고리즘 차이"와 "환경 초기화 차이"로 분리하기 쉬워짐
- 재실험 시 비교 신뢰도 향상

## 8. Align Experiment Entry Points

### Current Problem

`main.py`는 `real_ratio`, `uncertainty_threshold` override를 지원하지만, `run_multi_seed.py`는 기본 config만 사용한다.

이 차이 때문에 결과 디렉터리 구조가 일관되지 않고, 실험 폴더명이 혼재할 수 있다.

### Recommended Change

`run_multi_seed.py`에도 아래 인자를 추가한다.

- `--real-ratio`
- `--uncertainty-threshold`

그리고 `main.py`와 동일한 규칙으로 `run_name`을 생성한다.

### Example Patch Direction

```python
parser.add_argument('--real-ratio', type=float, default=None)
parser.add_argument('--uncertainty-threshold', type=float, default=None)
```

```python
cfg = DOBMBRLConfig()
if args.real_ratio is not None:
    cfg.real_ratio = args.real_ratio
if args.uncertainty_threshold is not None:
    cfg.uncertainty_threshold = args.uncertainty_threshold
```

### Expected Effect

- 실험 관리 쉬워짐
- 폴더명 혼선 감소
- 결과 재분석 자동화 쉬워짐

## Suggested Implementation Sequence

### Minimal Safe Patch Set

아래 4개는 같이 묶어서 들어가는 것을 추천한다.

1. `model_buffer.clear()`
2. `model_buffer.length` 검사 후 mixed batch 허용
3. RBF target/current calibration 재계산
4. conditional `effective_real_ratio`

이 조합은 기존 구조를 크게 바꾸지 않으면서도 가장 위험한 불안정성을 먼저 줄인다.

### Medium Patch Set

위 4개에 아래를 추가한다.

5. `h == 0` always-pass 완화
6. synthetic reward 개선

### Larger Experimental Patch Set

마지막으로 아래를 별도 실험으로 검증한다.

7. `res_net` 타깃을 `dhat`에서 direct residual로 변경

## Validation Plan

수정 후에는 아래 순서로 검증하는 것이 좋다.

### 1. Smoke Test

- single seed 50~100 episode 실행
- NaN 발생 여부 확인
- `model_buffer.length`가 의도대로 refresh되는지 확인
- `effective_real_ratio` 로그 확인

### 2. Stability Check

다음 설정으로 6 seeds 이상 비교:

- `real_ratio=1.0`
- `real_ratio=0.5, uncert_thresh=0.3`
- `real_ratio=0.2, uncert_thresh=0.3`

확인 지표:

- final reward
- best 10-episode moving average
- rollout pass rate
- rollout average horizon
- `rbf_calib_ratio`
- `rbf_calib_corr`
- critic `td_loss_avg`

### 3. Decision Criteria

아래 중 2개 이상 만족하면 개선으로 볼 수 있다.

- failing seed 비율 감소
- seed 간 reward 분산 감소
- `rbf_calib_ratio`가 1.0 근처로 개선
- `rbf_calib_corr` 안정적 상승
- reward가 좋아지면서 rollout horizon도 증가
- reward는 나쁜데 rollout pass rate만 높은 모순 패턴 감소

## Logging Additions Recommended

추가로 아래 로그를 CSV에 남기면 이후 분석이 쉬워진다.

- `effective_real_ratio`
- `model_buffer_used` 또는 `used_model_batch`
- `generated_model_samples`
- `model_buffer_length_before_update`
- `model_buffer_length_after_rollout`
- `rollout_first_step_pass_rate`
- `reward_model_loss` if reward model is introduced

## Final Recommendation

현재 구조에서 가장 먼저 손봐야 할 것은 "synthetic data를 얼마나 쉽게 critic 학습에 넣고 있는가"다.

따라서 1차 구현 우선순위는 아래와 같다.

1. `model_buffer` refresh
2. current `res_net` 기준 uncertainty 재계산
3. conditional model mixing
4. rollout gating 완화

그 다음 단계에서 synthetic reward와 residual target을 다듬는 것이 좋다.

이 순서로 가면 코드 변경 범위를 통제하면서도, 지금 관찰된 "어떤 시드는 잘되고 어떤 시드는 전혀 안 되는" 현상을 가장 직접적으로 줄일 가능성이 높다.
