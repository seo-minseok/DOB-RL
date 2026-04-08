# 02 — 코드 구조 가이드

## 패키지 구조 (base/ 또는 Cycle_N/ 내 동일)

```
dob_mbrl/
├── __init__.py
├── dynamics/
│   ├── constants.py       # 전역 상수 (OBS_MIN/MAX, FPINV, F_MAT, ACT_ELEMENTS 등)
│   ├── nominal.py         # default_cartpole_params, step_nominal_cartpole
│   ├── dob.py             # predict_next_obs_dob, compute_dob_update
│   └── __init__.py
├── models/
│   ├── q_network.py       # QNetwork
│   ├── residual_dx_net.py # ResidualDxNet
│   ├── normalized_rbf.py  # NormalizedRBFModel
│   └── __init__.py
├── training/
│   ├── config.py          # DOBMBRLConfig dataclass
│   ├── trainer.py         # train_DOB_core (메인 루프)
│   ├── model_learning.py  # train_residual_dx_model_dob, train_uncertainty_rbf
│   ├── rollout.py         # generate_samples_dob, sample_mixed_minibatch
│   └── __init__.py
├── envs/
│   ├── cartpole_utils.py  # make_cartpole_env, reset_env, step_env, reward_is_done_function
│   └── __init__.py
└── utils/
    ├── buffer.py          # ReplayBufferDOB
    └── __init__.py
```

## 모듈 임포트 규칙

- 상수는 항상 `dob_mbrl.dynamics.constants`에서 임포트.
- 모델은 `dob_mbrl.models`에서 임포트.
- 학습 로직은 `dob_mbrl.training`에서 임포트.
- 순환 임포트 금지: dynamics → (없음), models → dynamics, training → models+dynamics+envs+utils, envs → dynamics.

## 새 모듈 추가 위치 가이드

| 추가 항목 | 위치 |
|---|---|
| 새 RL 알고리즘 | `dob_mbrl/training/trainer_{algo}.py` + config 추가 |
| 새 환경 | `dob_mbrl/envs/{env}_utils.py` + 해당 constants |
| 새 네트워크 | `dob_mbrl/models/{name}.py` |
| 새 전역 상수 | `dob_mbrl/dynamics/constants.py` (또는 별도 파일) |

## 핵심 임포트 검증 명령

```bash
python -c "from dob_mbrl.models import QNetwork, ResidualDxNet, NormalizedRBFModel"
python -c "from dob_mbrl.utils.buffer import ReplayBufferDOB"
python -c "from dob_mbrl.training.trainer import train_DOB_core"
python main.py --help
```
