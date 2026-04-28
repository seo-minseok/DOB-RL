# base — 아카이브된 기본 프로젝트

> **상태: 아카이브 (수정 금지)**
> 생성일: 2026-04-08

이 디렉토리는 DOB-MBRL 프로젝트의 기본 구조가 아카이브된 상태이다.

## 절대 규칙

- **사용자가 명시적으로 "base 수정해줘"라고 지시하지 않는 한 이 디렉토리의 파일을 수정해서는 안 된다.**
- `original/` 내 원본 파일(`train_DOB_core.py`, `Multi_Seed_DOB_Exp.py`)은 **어떤 경우에도 수정 금지.**

## 역할

새 실험 Cycle의 시작점(소스)으로 사용된다. `create_cycle.sh --from base`가 이 디렉토리에서 복사한다.

## 구조

```
base/
├── original/              # 원본 코드 (읽기 전용)
│   ├── train_DOB_core.py
│   └── Multi_Seed_DOB_Exp.py
├── dob_mbrl/              # 재구성된 패키지
│   ├── dynamics/          # constants, nominal, dob
│   ├── models/            # QNetwork, ResidualDxNet, NormalizedRBFModel
│   ├── training/          # config, trainer, model_learning, rollout
│   ├── envs/              # cartpole_utils
│   └── utils/             # buffer
├── main.py                # 단일 시드 진입점
├── run_multi_seed.py      # 멀티 시드 진입점
└── scripts/
    └── plot_results.py    # 결과 시각화
```

## 임포트 검증

```bash
cd base
python -c "from dob_mbrl.models import QNetwork, ResidualDxNet, NormalizedRBFModel"
python -c "from dob_mbrl.utils.buffer import ReplayBufferDOB"
python -c "from dob_mbrl.training.trainer import train_DOB_core"
python main.py --help
```
