# DOB-MBRL: Disturbance Observer-Based Model-Based Reinforcement Learning

CartPole 환경에서 **Nominal Dynamics + Learned Residual Dynamics + Disturbance Observer(DOB)** 를 결합한 강화학습 프레임워크.

원본 MATLAB 구현을 PyTorch + Gymnasium 기반으로 이식하고, 반복 실험을 위한 **Cycle 기반 실험 관리 시스템**을 함께 제공한다.

---

## 알고리즘 개요

DOB-MBRL은 세 가지 구성 요소를 결합한다.

```
실제 환경 관측
        │
        ▼
┌───────────────────┐     ┌──────────────────────┐
│  Nominal Dynamics  │ +   │  Residual Dynamics Net │
│  (물리 방정식 기반) │     │  (잔차 신경망, 32-dim) │
└───────────────────┘     └──────────────────────┘
        │                           │
        └────────────┬──────────────┘
                     │
                     ▼
          ┌──────────────────┐
          │  DOB (온라인 보정) │  ← 모델 오차를 에피소드마다 실시간 추정
          └──────────────────┘
                     │
                     ▼
          ┌──────────────────┐
          │    Q-Network      │  ← DQN 계열, 128-128-2 FC
          └──────────────────┘
```

**핵심 아이디어**
- **Nominal Model**: 가속도를 0으로 단순화한 Euler 적분 모델 (빠른 계산)
- **ResidualDxNet**: Nominal이 설명하지 못하는 잔차(속도·각속도 변화)를 5-dim 입력(관측 4 + 힘 1)으로 학습
- **DOB**: `dhat = dob_w * dx_res + (1 - dob_w) * (FPINV @ e)` 로 온라인 보정 (에피소드마다 `zeros(2)` 리셋)
- **RBF Uncertainty**: Normalized RBF 모델로 rollout 신뢰도를 추정, 불확실한 영역에서 rollout 조기 종료
- **Model Buffer**: 가상 rollout으로 샘플 효율 향상 (`real_ratio=0.2` → 실제 20% + 가상 80%)

---

## 프로젝트 구조

```
DOB-RL/
├── base/                          # 기본 코드 아카이브 (수정 금지)
│   ├── original/                  # 원본 파일 (절대 수정 금지)
│   │   ├── train_DOB_core.py
│   │   └── Multi_Seed_DOB_Exp.py
│   ├── dob_mbrl/                  # 모듈화된 패키지
│   ├── main.py
│   ├── run_multi_seed.py
│   └── scripts/
│       └── plot_results.py
│
├── cycles/                        # 실험 Cycle 관리
│   └── Cycle_N/                   # 각 실험 단위
│       ├── CYCLE_REPORT.md        # 실험 보고서
│       ├── dob_mbrl/              # base에서 복사된 코드 (수정 가능)
│       ├── main.py
│       ├── run_multi_seed.py
│       ├── scripts/
│       ├── checkpoints/           # 학습된 모델 저장
│       ├── figures/               # 시각화 결과
│       ├── logs/                  # 단일 시드 로그
│       └── results/               # 멀티 시드 결과
│
├── .claude/
│   ├── instructions/              # 에이전트 작업 지침서
│   └── skills/                    # 자동화 스킬 스크립트
│       ├── cycle-manager/         # Cycle 생성 자동화
│       └── plot-generator/        # Figure 생성 자동화
│
├── CLAUDE.md                      # 에이전트 절대 규칙
├── memory.md                      # 에이전트 지식 메모리
├── experiments_log.md             # Cycle별 실험 이력 테이블
├── dob_mbrl_agent_design.md       # 시스템 설계 문서
└── requirements.txt
```

---

## 패키지 구조 (`dob_mbrl/`)

```
dob_mbrl/
├── dynamics/
│   ├── constants.py       # 전역 상수 (관측 범위, ACT_ELEMENTS, FPINV, F_MAT)
│   ├── nominal.py         # Nominal CartPole Dynamics (Euler 적분)
│   └── dob.py             # DOB 관련 유틸리티
├── models/
│   ├── q_network.py       # QNetwork — FC(128)→ReLU→FC(128)→ReLU→FC(2)
│   ├── residual_dx_net.py # ResidualDxNet — FC(32)→ReLU→FC(32)→ReLU→FC(2)
│   └── normalized_rbf.py  # NormalizedRBFModel — 불확실성 추정
├── training/
│   ├── config.py          # DOBMBRLConfig (dataclass, 하이퍼파라미터 전체)
│   ├── trainer.py         # train_DOB_core — 메인 학습 루프
│   ├── model_learning.py  # Residual 및 RBF 모델 학습 함수
│   └── rollout.py         # 가상 rollout 생성 및 mixed 미니배치 샘플링
├── envs/
│   └── cartpole_utils.py  # make_cartpole_env, reset_env, step_env (5-tuple 처리)
└── utils/
    └── buffer.py          # ReplayBufferDOB (real_buffer, model_buffer 분리 운영)
```

---

## 설치

```bash
pip install -r requirements.txt
```

**요구 사항**

| 패키지 | 최소 버전 |
|---|---|
| torch | ≥ 2.0.0 |
| gymnasium | ≥ 0.29.0 |
| numpy | ≥ 1.24.0 |
| matplotlib | ≥ 3.7.0 |

CUDA가 설치되어 있으면 `torch+cu*` 버전 사용을 권장한다.

---

## 학습 실행

모든 학습 명령은 해당 Cycle 디렉토리 기준으로 실행한다.

### 단일 시드

```bash
cd cycles/Cycle_1
python main.py --checkpoint-dir ./checkpoints --seed 1
```

| 옵션 | 설명 |
|---|---|
| `--checkpoint-dir` | 체크포인트 저장 경로 (기본: `./checkpoints`) |
| `--seed` | 랜덤 시드 번호 (기본: 1) |
| `--resume` | 기존 체크포인트에서 재개 |

로그: `cycles/Cycle_N/logs/seed_{N}_result.pkl`

### 멀티 시드 병렬

```bash
cd cycles/Cycle_1
python run_multi_seed.py --checkpoint-dir ./checkpoints --num-seeds 16
```

| 옵션 | 설명 |
|---|---|
| `--checkpoint-dir` | 체크포인트 저장 경로 |
| `--num-seeds` | 병렬 실행 시드 수 (기본: 16) |

워커 수는 `min(num_seeds, cpu_count())`로 자동 결정된다.

결과: `cycles/Cycle_N/results/DOB_MBRL_MultiSeed_Result.pkl`

### Checkpoint 저장 조건

최근 10 에피소드 평균 reward ≥ 480 이고 기존 최고 기록 초과 시 저장.

```
cycles/Cycle_N/checkpoints/Champion_Seed{seed}_BestModel.pt
```

---

## 하이퍼파라미터 수정

`cycles/Cycle_N/dob_mbrl/training/config.py`를 **직접 편집**한다. CLI override는 없다.

```python
@dataclass
class DOBMBRLConfig:
    num_episodes: int        = 200       # 총 에피소드
    max_steps_per_ep: int    = 500       # 에피소드 최대 스텝
    warm_start_samples: int  = 200       # 탐색 시작 전 수집 샘플 수

    lr_critic: float         = 1e-3      # Q-Network Adam lr
    discount_factor: float   = 0.99
    tau: float               = 0.005     # Target network soft-update

    epsilon: float           = 1.0       # 초기 탐색률
    epsilon_min: float       = 0.01
    epsilon_decay: float     = 0.005

    real_ratio: float        = 0.2       # Mixed 배치에서 실제 데이터 비율
    dob_w: float             = 0.1       # DOB 가중치

    num_rbf_centers: int     = 600       # RBF 모델 센터 수
    ...
```

수정 후 검증:

```bash
python -c "from dob_mbrl.training.config import DOBMBRLConfig; DOBMBRLConfig()"
```

---

## 결과 시각화

```bash
# 단일 시드
cd cycles/Cycle_1
python scripts/plot_results.py --log-dir ./logs --seed 1 --figures-dir ./figures

# 멀티 시드
python scripts/plot_results.py --results-dir ./results --multi-seed --figures-dir ./figures
```

**생성 파일**

| 파일 | 내용 |
|---|---|
| `reward_curve.png` | 에피소드별 누적 reward |
| `reward_smoothed.png` | 10 에피소드 이동 평균 reward |
| `total_steps.png` | 누적 환경 스텝 |
| `multiseed_mean_std.png` | 멀티 시드 mean ± std (환경 스텝 기준) |

---

## Cycle 기반 실험 관리

실험 변경 사항은 **항상 새 Cycle 폴더에서만** 수행한다. `base/`는 절대 수정하지 않는다.

### 새 Cycle 생성

```bash
# base에서 시작
bash .claude/skills/cycle-manager/scripts/create_cycle.sh --from base

# 이전 Cycle에서 시작
bash .claude/skills/cycle-manager/scripts/create_cycle.sh --from cycles/Cycle_1
```

Cycle 번호는 `cycles/` 내 기존 최대 번호 + 1로 자동 부여된다.

### Cycle 폴더 구조

```
cycles/
├── Cycle_1/    ← 기초 검증 (base 그대로)
├── Cycle_2/    ← 실험 A (예: GPU 지원 추가)
└── Cycle_3/    ← 실험 B (예: 하이퍼파라미터 튜닝)
```

각 Cycle은 독립적으로 실행되며, 이전 Cycle 결과에 영향을 주지 않는다.

---

## 실험 이력

`experiments_log.md`에 Cycle별 실험 이력이 기록된다.

---

## 주요 설계 원칙

1. **`base/` 수정 금지** — 사용자가 명시적으로 허가한 경우에만 가능
2. **`base/original/` 절대 수정 금지** — 원본 MATLAB 이식본, 참조용 읽기만 허용
3. **모든 코드 변경은 `cycles/Cycle_N/` 내에서만**
4. **1 figure = 1 파일** — `plt.subplots()`는 항상 단일 Axes `(1, 1)`, subplot 금지
5. **`config.py` 수정은 파일 직접 편집** — CLI override 없음
