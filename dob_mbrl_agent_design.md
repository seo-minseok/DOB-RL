# DOB-MBRL 강화학습 프로젝트 에이전트 시스템 설계서

> **문서 목적**: Claude Code에서 프로젝트 구현 시 참조하는 계획서
> **작성일**: 2026-04-08
> **상태**: 인터뷰 완료 / 구현 준비

---

## 1. 작업 컨텍스트

### 1.1 배경

DOB-MBRL(Disturbance Observer-Based Model-Based Reinforcement Learning)은 CartPole 환경에서 nominal dynamics와 learned residual dynamics를 결합하고, Disturbance Observer로 모델 오차를 온라인 보정하는 강화학습 알고리즘이다. 기존에 MATLAB으로 구현된 코드를 Codex를 통해 PyTorch+Gymnasium으로 1:1 변환한 상태이며, 현재 `train_DOB_core.py`(학습 코어)와 `Multi_Seed_DOB_Exp.py`(멀티시드 실험 러너) 두 파일로 구성되어 있다.

### 1.2 목적

1. **코드 정리/가독성**: 모놀리식 두 파일을 일반적인 RL 라이브러리 구조로 재구성 (기능 변경 없이 구조만 분리)
2. **확장성 확보**: 새로운 RL 알고리즘, 환경, 네트워크 구조를 쉽게 추가할 수 있는 모듈화된 아키텍처
3. **Cycle 기반 실험 관리**: 기본 프로젝트를 아카이브하고, 변경 사항은 독립적인 Cycle 폴더에서만 수행
4. **실험 이력 추적**: Cycle 간 비교와 진행 상황을 한눈에 파악할 수 있는 상위 레이어 문서 유지

### 1.3 범위

- **포함**: 프로젝트 폴더 구조 재설계, Cycle 관리 워크플로우, 에이전트 지침 체계, 메모리 시스템, 플로팅 규칙, requirements.txt, .gitignore, experiments_log.md, resume 로직
- **제외**: 알고리즘 자체의 수정, 새로운 환경/알고리즘 구현 (이는 향후 Cycle에서 수행), docs/algorithm_reference.md

### 1.4 입출력 정의

| 구분 | 내용 |
|------|------|
| **입력** | 기존 `train_DOB_core.py`, `Multi_Seed_DOB_Exp.py` (읽기 전용 참조) |
| **출력** | 재구성된 프로젝트 구조 + CLAUDE.md 지침 체계 + Cycle 관리 시스템 + requirements.txt + .gitignore + experiments_log.md |

### 1.5 제약조건

- `train_DOB_core.py`, `Multi_Seed_DOB_Exp.py` 원본 파일은 **절대 수정 금지** (참조용으로만 보관)
- `base/` 디렉토리 전체는 **절대 수정 금지** — CLAUDE.md에 명시적 절대 금지 규칙 기재
- 각 Cycle은 기본적으로 base에서 시작하되, 사용자가 이전 Cycle 지정 시 해당 Cycle에서 복사 (에이전트가 구두 확인 후 실행)
- Figure 출력 규칙: **하나의 이미지 파일에는 반드시 하나의 figure만** (subplot 금지)
  - 단, Cycle 간 비교용 오버레이 그래프(mean±std 중첩)는 subplot이 아니므로 허용
- config.py 수정은 파일 직접 편집만 허용 (CLI override 없음)

### 1.6 용어 정의

| 용어 | 정의 |
|------|------|
| **DOB** | Disturbance Observer — 모델 예측과 실제 dynamics 간 오차를 온라인으로 추정·보정하는 관측기. `dhat`은 **에피소드마다 리셋** (원본 코드 동작 그대로) |
| **MBRL** | Model-Based Reinforcement Learning — 환경 모델을 학습하여 가상 rollout으로 데이터 효율을 높이는 RL |
| **Residual Dynamics** | Nominal model이 설명하지 못하는 잔차를 신경망으로 학습. **Rollout 중 동결** (모델 업데이트는 에피소드 시작 전 Phase 1에서만 수행) |
| **RBF Uncertainty** | Normalized RBF 기반 불확실성 추정 모델, rollout 신뢰도 판단에 사용. 첫 번째 rollout step(h==0)에서는 항상 신뢰 가능으로 처리 |
| **real_ratio** | sample_mixed_minibatch에서 real 데이터 비율. 기본값 0.2 (20% real, 80% model). config.py에서 변경 가능 |
| **Cycle** | 하나의 독립적 실험 단위. 기본 프로젝트(base 또는 지정 Cycle)를 복사하여 해당 폴더 내에서만 코드 수정 |
| **Base** | 최초 구성된 기본 프로젝트 구조. 아카이브 후 Cycle의 원본으로 사용. **사용자 명시 지시 없이 절대 수정 금지** |

---

## 2. 워크플로우 정의

### 2.1 전체 워크플로우 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    에이전트 작업 워크플로우                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [시작] → 메모리 파일 읽기(memory.md) ← 모든 작업 시작 시 필수  │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────┐     ┌─────────────────────────────┐   │
│  │ A. 프로젝트 초기화   │     │ B. 실험 Cycle 수행           │   │
│  │   (최초 1회)         │     │   (반복)                     │   │
│  └──────────┬──────────┘     └──────────┬──────────────────┘   │
│             │                           │                       │
│             ▼                           ▼                       │
│   Base 프로젝트 구조 생성      에이전트 구두 확인:               │
│             │                  "base에서? Cycle_N에서?"         │
│             ▼                           │                       │
│   원본 코드를 모듈별 분리       create_cycle.sh 실행            │
│             │                           │                       │
│             ▼                           ▼                       │
│   구조 검증 (임포트 테스트)     요청된 코드 수정 수행             │
│             │                           │                       │
│             ▼                           ▼                       │
│   Base 아카이브 완료           학습 실행                        │
│                                         │                       │
│                                         ▼                       │
│                                결과 분석 + figure 생성           │
│                                         │                       │
│                                         ▼                       │
│                                Cycle 보고서 작성 (에이전트)      │
│                                         │                       │
│                                         ▼                       │
│                                experiments_log.md 업데이트      │
│                                         │                       │
│                                         ▼                       │
│                                메모리 업데이트 제안               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 워크플로우 A: 프로젝트 초기화 (최초 1회)

#### 단계 A1: 원본 코드 분석

- **수행 주체**: 에이전트 (LLM 판단)
- **동작**: `train_DOB_core.py`와 `Multi_Seed_DOB_Exp.py`를 읽고 모듈 분리 계획 수립
- **성공 기준**: 각 클래스/함수가 어떤 모듈로 배치될지 매핑 완료
- **검증 방법**: LLM 자기 검증 (매핑 누락 없는지 확인)
- **실패 시 처리**: 자동 재시도 (최대 2회) — 누락 항목 재확인

#### 단계 A2: Base 프로젝트 구조 생성

- **수행 주체**: 에이전트 (코드 처리)
- **동작**: 설계서의 폴더 구조에 따라 디렉토리 및 파일 생성. requirements.txt와 .gitignore도 함께 생성
- **성공 기준**: 모든 디렉토리 존재, `__init__.py` 파일 정상, 원본 코드 기능 100% 보존
- **검증 방법**: 스키마 검증 (디렉토리 구조 확인) + 규칙 기반 (임포트 테스트 실행)
- **실패 시 처리**: 자동 재시도 (최대 3회) — 임포트 에러 수정 후 재검증

#### 단계 A3: 기능 검증

- **수행 주체**: 스크립트 (자동)
- **동작**: `python -c "from dob_mbrl.models import QNetwork, ResidualDxNet, NormalizedRBFModel"` 등 핵심 임포트 테스트
- **성공 기준**: 모든 임포트 성공, 에러 0건
- **검증 방법**: 규칙 기반 (exit code 0)
- **실패 시 처리**: 자동 재시도 (최대 3회) — 에러 메시지 기반 수정

#### 단계 A4: Base 아카이브

- **수행 주체**: 에이전트 (코드 처리)
- **동작**: Base 프로젝트 완성 후 아카이브 상태로 전환 (이후 사용자 명시 지시 없이는 수정 금지)
- **성공 기준**: Base 디렉토리 내 모든 파일 정상, README에 아카이브 상태 명시
- **검증 방법**: 스키마 검증
- **실패 시 처리**: 에스컬레이션 — 사용자에게 확인 요청

### 2.3 워크플로우 B: 실험 Cycle 수행 (반복)

#### 단계 B1: Cycle 폴더 생성

- **수행 주체**: 에이전트 (구두 확인) + 스크립트 (create_cycle.sh)
- **동작**:
  1. 에이전트가 사용자에게 확인: "base에서 시작할까요, 아니면 특정 Cycle(예: Cycle_2)에서 시작할까요?"
  2. 확인 후 `create_cycle.sh --from base` 또는 `create_cycle.sh --from Cycle_N` 실행
  3. Cycle 번호는 기존 Cycle 중 최대 번호 + 1 자동 부여
- **성공 기준**: 복사된 디렉토리 구조가 소스(base 또는 지정 Cycle)와 동일
- **검증 방법**: 규칙 기반 (파일 수 비교, diff 확인)
- **실패 시 처리**: 자동 재시도 (최대 2회)

> **Cycle 시작점 정책**
> - 기본값: base에서 복사
> - 사용자가 "Cycle_N에서 시작" 명시 시: 해당 Cycle의 코드만 복사 (checkpoints/logs/figures는 복사 안 함)
> - 실패한 Cycle의 코드 계승 여부: 사람이 명시적으로 지정 ("Cycle_2 코드 기반으로 Cycle_5 만들어줘")

#### 단계 B2: 코드 수정

- **수행 주체**: 에이전트 (LLM 판단 + 코드 처리)
- **동작**: 사용자가 요청한 변경 사항을 `Cycle_N/` 내에서만 수행
- **분기 조건**:
  - 변경 사항이 Base 수정을 필요로 하는 경우 → **에스컬레이션** (사용자에게 Base 수정 허가 요청)
  - 변경 사항이 Cycle 범위 내인 경우 → 바로 진행
- **성공 기준**: 요청된 변경 사항 반영 완료, 기존 기능 미파괴
- **검증 방법**: 규칙 기반 (임포트 테스트) + LLM 자기 검증 (변경 의도 충족 여부)
- **실패 시 처리**: 자동 재시도 (최대 3회)

#### 단계 B3: 학습 실행

- **수행 주체**: 사용자 (명령 직접 실행)
- **동작**: 에이전트가 정확한 실행 명령을 안내. 학습 실행은 사용자가 직접 수행
- **CLI 스펙**:
  ```bash
  # 단일 시드 학습
  python main.py --checkpoint-dir ./checkpoints --seed 0

  # 멀티 시드 학습
  python run_multi_seed.py --checkpoint-dir ./checkpoints --num-seeds 16
  ```
- **Resume 지원**: `--resume` 인자로 마지막 checkpoint에서 이어서 학습
  ```bash
  python main.py --checkpoint-dir ./checkpoints --seed 0 --resume
  ```
- **성공 기준**: 학습 완료, 모델 체크포인트 저장, 로그 파일 생성
- **검증 방법**: 규칙 기반 (출력 파일 존재 확인)
- **실패 시 처리**: 에스컬레이션 — 에러 로그를 사용자에게 보고

#### 단계 B4: 결과 분석 및 Figure 생성

- **수행 주체**: 스크립트 (자동) + 에이전트 (LLM 판단)
- **동작**: 학습 로그를 파싱하여 figure 생성. **하나의 이미지 파일 = 하나의 figure** 규칙 엄수
- **Cycle 간 비교 figure**: 서로 다른 설정의 mean±std를 하나의 figure에 오버레이하는 것은 허용 (subplot이 아님)
- **성공 기준**: 모든 로그 metric에 대해 개별 figure 파일 생성
- **검증 방법**: 규칙 기반 (figure 파일 수 = metric 수, 각 파일에 subplot 없음)
- **실패 시 처리**: 자동 재시도 (최대 2회) — subplot 감지 시 분리

#### 단계 B5: Cycle 요약 보고서 작성

- **수행 주체**: 에이전트 (거의 전부 자동 생성)
- **동작**: `Cycle_N/CYCLE_REPORT.md` 작성. 에이전트가 수집한 정보(코드 변경 내역, 수치, figure 경로)를 자동 입력하고, "관찰 및 교훈" 섹션만 빈 섹션으로 예약 (사람이 나중에 채움)
- **성공 기준**: 필수 섹션 모두 포함 (변경점, 하이퍼파라미터, 학습 결과, figure 목록, 관찰/교훈 예약 섹션)
- **검증 방법**: 스키마 검증 (필수 섹션 존재) + LLM 자기 검증 (내용 누락 확인)
- **실패 시 처리**: 자동 재시도 (최대 2회)

#### 단계 B6: experiments_log.md 업데이트

- **수행 주체**: 에이전트 (자동)
- **동작**: 프로젝트 root의 `experiments_log.md`에 이번 Cycle 항목 추가 (Cycle 번호, 날짜, 핵심 변경점, 최종 metric 한 줄 요약)
- **성공 기준**: experiments_log.md에 신규 항목 추가 완료

#### 단계 B7: 메모리 업데이트 제안

- **수행 주체**: 에이전트 (LLM 판단)
- **동작**: 이번 Cycle에서 발견된 실수, 주의사항, 패턴을 `memory.md`에 추가할 후보로 제안
- **분기 조건**:
  - 사용자 승인 → `memory.md`에 추가
  - 사용자 거부 → 스킵 (재제안 없음)
- **성공 기준**: 제안 내용이 구체적이고 재현 가능한 교훈
- **검증 방법**: 사람 검토
- **실패 시 처리**: 스킵

### 2.4 Base 수정 전파 정책

사용자가 Base 수정을 승인하면:

1. 에이전트가 Base 파일 수정 수행
2. Base 임포트 테스트 재실행
3. **기존 Cycle은 영향 없음** — 이미 완료된 Cycle은 수정 전 Base 기반으로 동작
4. **신규 Cycle부터 새 Base 적용** — 이후 `create_cycle.sh`는 변경된 Base에서 복사

### 2.5 LLM 판단 vs 코드 처리 구분

| 에이전트가 직접 수행 | 스크립트로 처리 |
|---|---|
| 모듈 분리 계획 수립 | Cycle 폴더 복사 (create_cycle.sh) |
| 코드 변경 사항 해석 및 구현 | 임포트 테스트 실행 |
| Cycle 보고서 작성 | 학습 로그 파싱 |
| 메모리 업데이트 후보 판단 | Figure 생성 (matplotlib) |
| Base 수정 필요 여부 판단 | 모델 체크포인트 저장 |
| 에러 원인 분석 및 수정 방향 결정 | 디렉토리 구조 검증 |
| Cycle 시작점 사용자 확인 | experiments_log.md 업데이트 |

---

## 3. 구현 스펙

### 3.1 프로젝트 폴더 구조

```
/project-root
├── CLAUDE.md                              # 메인 에이전트 지침 (라우터 + 조건부 로딩 규칙)
├── memory.md                              # 공용 메모리 (에이전트 참조, 사람이 승인)
├── experiments_log.md                     # Cycle 간 비교 및 진행 상황 요약 (에이전트 자동 업데이트)
├── requirements.txt                       # 의존성 고정 (프로젝트 초기화 시 생성)
├── .gitignore                             # checkpoints/, logs/, figures/, results/ 제외
│
├── /.claude
│   ├── /instructions                      # 세분화된 지시서들
│   │   ├── 01_project-rules.md            # 프로젝트 전반 규칙 (Base 보호, Cycle 규칙)
│   │   ├── 02_code-structure.md           # 코드 구조 가이드 (모듈 배치, 임포트 규칙)
│   │   ├── 03_cycle-management.md         # Cycle 생성/관리 절차
│   │   ├── 04_training.md                 # 학습 실행 관련 지침 (CLI 인자, resume 로직)
│   │   ├── 05_plotting.md                 # Figure 생성 규칙 (1 figure per file, Cycle 간 비교 허용)
│   │   ├── 06_reporting.md                # Cycle 보고서 작성 가이드
│   │   └── 07_memory-protocol.md          # 메모리 업데이트 프로토콜
│   │
│   ├── /skills/cycle-manager
│   │   ├── SKILL.md                       # Cycle 생성/복사 스킬 정의 (Claude Code 공식 형식)
│   │   └── /scripts
│   │       └── create_cycle.sh            # --from base 또는 --from Cycle_N 인자 지원
│   │
│   └── /skills/plot-generator
│       ├── SKILL.md                       # 플로팅 스킬 정의 (Claude Code 공식 형식)
│       └── /scripts
│           └── plot_metrics.py            # 로그 → 개별 figure 변환
│
├── /base                                  # ★ 아카이브된 기본 프로젝트 (절대 수정 금지)
│   ├── README.md                          # 아카이브 상태 명시
│   ├── /original                          # 원본 코드 보관 (읽기 전용 참조, 절대 수정 금지)
│   │   ├── train_DOB_core.py
│   │   └── Multi_Seed_DOB_Exp.py
│   │
│   ├── /dob_mbrl                          # 재구성된 패키지
│   │   ├── __init__.py
│   │   ├── /envs
│   │   │   ├── __init__.py
│   │   │   └── cartpole_utils.py          # make_cartpole_env, reset_env, step_env, reward_is_done_function
│   │   ├── /models
│   │   │   ├── __init__.py
│   │   │   ├── q_network.py               # QNetwork
│   │   │   ├── residual_dx_net.py         # ResidualDxNet
│   │   │   └── normalized_rbf.py          # NormalizedRBFModel
│   │   ├── /training
│   │   │   ├── __init__.py
│   │   │   ├── config.py                  # 하이퍼파라미터 dataclass + __post_init__ 검증
│   │   │   ├── trainer.py                 # train_DOB_core 메인 루프 래핑 + resume 지원
│   │   │   ├── model_learning.py          # train_residual_dx_model_dob, train_uncertainty_rbf
│   │   │   └── rollout.py                 # generate_samples_dob, sample_mixed_minibatch
│   │   ├── /dynamics
│   │   │   ├── __init__.py
│   │   │   ├── nominal.py                 # default_cartpole_params, step_nominal_cartpole
│   │   │   ├── dob.py                     # DOB 온라인 업데이트 로직, predict_next_obs_dob
│   │   │   └── constants.py               # FPINV, F_MAT, OBS_MIN/MAX, ACT_ELEMENTS 등
│   │   └── /utils
│   │       ├── __init__.py
│   │       └── buffer.py                  # ReplayBufferDOB (real_buffer / model_buffer 두 개 분리 운영)
│   │
│   ├── main.py                            # 단일 시드 학습 진입점 (checkpoint_dir, seed, resume 인자)
│   ├── run_multi_seed.py                  # 멀티 시드 실험 진입점 (checkpoint_dir, num_seeds 인자)
│   └── /scripts
│       └── plot_results.py                # 결과 시각화 (1 figure per file 규칙)
│
├── /Cycle_1                               # 첫 번째 실험 Cycle (base 복사본)
│   ├── CYCLE_REPORT.md                    # 이 Cycle의 요약 보고서
│   ├── /dob_mbrl/...                      # (base와 동일 구조, 여기서만 수정)
│   ├── /checkpoints                       # 학습된 모델 저장 (git 제외)
│   │   └── Champion_Seed{N}_BestModel.pt
│   ├── /figures                           # 학습 figure 저장 (git 제외)
│   │   ├── reward_curve.png
│   │   ├── epsilon_decay.png
│   │   └── ...
│   ├── /logs                              # 학습 로그 pkl (git 제외)
│   └── /results                           # 최종 결과 (멀티시드 집계 등, git 제외)
│
├── /Cycle_2
│   └── ...
│
└── (docs/ 는 이번 범위에서 제외)
```

### 3.2 원본 코드 → 모듈 매핑

| 원본 위치 (train_DOB_core.py) | 재구성 후 위치 | 비고 |
|---|---|---|
| 상수 (OBS_MIN/MAX, FPINV, F_MAT, ACT_ELEMENTS 등) | `dob_mbrl/dynamics/constants.py` | 전역 상수 중앙 관리 |
| `default_cartpole_params()` | `dob_mbrl/dynamics/nominal.py` | |
| `step_nominal_cartpole()` | `dob_mbrl/dynamics/nominal.py` | |
| `QNetwork` | `dob_mbrl/models/q_network.py` | |
| `ResidualDxNet` | `dob_mbrl/models/residual_dx_net.py` | |
| `NormalizedRBFModel` | `dob_mbrl/models/normalized_rbf.py` | |
| `ReplayBufferDOB` | `dob_mbrl/utils/buffer.py` | real/model 각 1개씩 인스턴스화 |
| `reward_is_done_function()` | `dob_mbrl/envs/cartpole_utils.py` | |
| `make_cartpole_env()`, `reset_env()`, `step_env()` | `dob_mbrl/envs/cartpole_utils.py` | |
| `predict_next_obs_dob()` | `dob_mbrl/dynamics/dob.py` | |
| DOB 온라인 업데이트 로직 (`dhat` 계산) | `dob_mbrl/dynamics/dob.py` | `dhat`은 에피소드마다 리셋 |
| `train_residual_dx_model_dob()` | `dob_mbrl/training/model_learning.py` | |
| `train_uncertainty_rbf()` | `dob_mbrl/training/model_learning.py` | |
| `generate_samples_dob()` | `dob_mbrl/training/rollout.py` | |
| `sample_mixed_minibatch()` | `dob_mbrl/training/rollout.py` | `real_ratio`는 config에서 주입 |
| `train_DOB_core()` 메인 루프 | `dob_mbrl/training/trainer.py` | checkpoint_dir 인자로 저장 경로 주입, resume 지원 추가 |
| 하이퍼파라미터 블록 | `dob_mbrl/training/config.py` | dataclass + `__post_init__` 검증 |

| 원본 위치 (Multi_Seed_DOB_Exp.py) | 재구성 후 위치 | 비고 |
|---|---|---|
| `_run_single()` + 메인 블록 | `run_multi_seed.py` | |
| 보간/집계 로직 | `run_multi_seed.py` | |
| 시각화 로직 | `scripts/plot_results.py` | 분리하여 별도 실행 가능 |

### 3.3 CLAUDE.md 핵심 섹션 목록

CLAUDE.md는 **라우터** 역할을 수행하며, **조건부 지시서 로딩 규칙**을 명시한다.

```
CLAUDE.md 구성:
1. 프로젝트 개요 (1-2문장)
2. 필수 선행 작업: memory.md 읽기 (모든 작업 시작 전)
3. 절대 규칙 (인라인):
   - base/ 디렉토리는 절대 수정 금지 (사용자 명시 지시 있을 때만 허용)
   - 1 figure per file (subplot 금지, Cycle 간 오버레이는 허용)
4. 조건부 지시서 로딩 규칙:
   - Base 구조 수정 / 코드 임포트 관련 → 02_code-structure.md
   - Cycle 생성 요청 → 03_cycle-management.md
   - 학습 실행 / checkpoint / resume → 04_training.md
   - Figure 생성 → 05_plotting.md
   - 보고서 작성 → 06_reporting.md
   - memory.md 수정 → 07_memory-protocol.md
   - Base 보호 규칙 확인 → 01_project-rules.md
5. 지시서 참조 목록 (각 파일의 역할과 경로)
```

#### 지시서별 핵심 내용

| 파일 | 역할 |
|---|---|
| `01_project-rules.md` | Base 보호 절대 규칙, Cycle 내에서만 작업, 원본 파일 수정 금지. 각 지시서 간 도메인이 달라 충돌 없도록 설계 |
| `02_code-structure.md` | 패키지 구조, 모듈 간 임포트 규칙, 새 모듈 추가 시 위치 가이드 |
| `03_cycle-management.md` | Cycle 생성 절차 (에이전트 구두 확인 → create_cycle.sh), 번호 부여, 복사 범위, CYCLE_REPORT.md 템플릿, Base 수정 전파 정책 |
| `04_training.md` | 학습 실행 방법 (CLI 스펙), checkpoint 저장 위치 (main.py에서 주입), resume 로직, 로그 포맷 |
| `05_plotting.md` | 1 figure per file 규칙, Cycle 간 오버레이 허용 조건, 파일 명명 규칙, figure 저장 경로 |
| `06_reporting.md` | CYCLE_REPORT.md 필수 섹션, 에이전트 자동 생성 범위, 관찰/교훈 섹션 빈 예약 방식, experiments_log.md 업데이트 규칙 |
| `07_memory-protocol.md` | memory.md 읽기(모든 작업 시작 전) / 쓰기 프로토콜, 제안 → 승인 흐름, 거부 시 재제안 없음 |

### 3.4 에이전트 구조

**단일 에이전트** 구조를 채택한다.

**근거**: 워크플로우가 순차적이고 도메인 지식이 하나(RL/DOB-MBRL)로 통일되어 있으며, 지시서를 7개 파일로 분리하여 필요 시에만 참조하므로 컨텍스트 윈도우 부담이 크지 않다. 서브에이전트 간 조율 오버헤드를 피한다.

### 3.5 스킬/스크립트 파일 목록

#### 스킬 1: `cycle-manager`

| 항목 | 내용 |
|---|---|
| **역할** | 새 Cycle 폴더 생성 (Base 또는 지정 Cycle 복사) |
| **트리거 조건** | 사용자가 "새 Cycle 시작", "Cycle_N 만들어줘" 등 요청 시. 에이전트가 구두로 시작점 확인 후 실행 |
| **입력** | `--from base` 또는 `--from Cycle_N` |
| **출력** | `Cycle_N/` 디렉토리 (소스 복사본 + 빈 checkpoints/figures/logs/results 폴더) |
| **스크립트** | `create_cycle.sh` — CLI 인자로 시작점 지정, 번호 자동 부여 |

#### 스킬 2: `plot-generator`

| 항목 | 내용 |
|---|---|
| **역할** | 학습 로그에서 metric별 개별 figure 생성 |
| **트리거 조건** | 학습 완료 후 결과 분석 요청 시, 또는 "figure 그려줘" 요청 시 |
| **입력** | 로그 파일 경로 (pkl) |
| **출력** | `Cycle_N/figures/` 하위에 metric별 개별 PNG 파일 |
| **스크립트** | `plot_metrics.py` — 로그를 파싱하여 metric당 하나의 figure 생성 |
| **핵심 규칙** | `plt.subplots()` 호출 시 반드시 `(1, 1)` 또는 단일 Axes. subplot 감지 시 에러 |

### 3.6 데이터 전달 패턴

| 단계 간 전달 | 방식 | 형식 |
|---|---|---|
| 학습 → 결과 분석 | 파일 기반 | `Cycle_N/logs/*.pkl` (`all_rewards`, `all_steps` 키) |
| 결과 분석 → figure | 파일 기반 | `Cycle_N/figures/*.png` |
| 학습 → 체크포인트 | 파일 기반 | `Cycle_N/checkpoints/Champion_Seed{N}_BestModel.pt` |
| Cycle 결과 → 보고서 | 에이전트 인라인 | figure 경로 + 수치 요약을 보고서에 직접 기술 |
| Cycle 보고서 → experiments_log | 에이전트 인라인 | 한 줄 요약 추가 |
| 에이전트 → 메모리 | 에이전트 인라인 | 제안 내용을 대화에서 사용자에게 제시 |

### 3.7 주요 산출물 파일 형식

| 산출물 | 형식 | 위치 | Git 추적 |
|---|---|---|---|
| 모델 체크포인트 | `.pt` (PyTorch state_dict) | `Cycle_N/checkpoints/` | ❌ (.gitignore) |
| 학습 로그 | `.pkl` (pickle) | `Cycle_N/logs/` | ❌ (.gitignore) |
| 학습 Figure | `.png` (개별 파일) | `Cycle_N/figures/` | ❌ (.gitignore) |
| 멀티시드 집계 결과 | `.pkl` + `.png` | `Cycle_N/results/` | ❌ (.gitignore) |
| Cycle 보고서 | `.md` (마크다운) | `Cycle_N/CYCLE_REPORT.md` | ✅ |
| 프로젝트 메모리 | `.md` (마크다운) | `/memory.md` | ✅ |
| 실험 이력 | `.md` (마크다운) | `/experiments_log.md` | ✅ |
| 코드 파일 | `.py` | `Cycle_N/dob_mbrl/`, `Cycle_N/main.py` 등 | ✅ |

### 3.8 CYCLE_REPORT.md 필수 섹션

```markdown
# Cycle N 실험 보고서

> 생성일: YYYY-MM-DD
> 시작점: base / Cycle_N

## 1. 변경점 요약
<!-- 에이전트 자동 생성: 이 Cycle에서 Base 대비 무엇을 변경했는가 -->

## 2. 변경 상세
<!-- 에이전트 자동 생성: 수정된 파일 목록 및 각 파일의 변경 내용 -->

## 3. 하이퍼파라미터
<!-- 에이전트 자동 생성: 이 Cycle에서 사용된 주요 하이퍼파라미터 값 -->

## 4. 학습 결과
<!-- 에이전트 자동 생성: 정량 지표 (최종 reward, 수렴 에피소드, 학습 시간 등) -->
<!-- 에이전트 자동 생성: Figure 파일 경로 목록 -->

## 5. 관찰 및 교훈
<!-- ★ 사람이 직접 작성 — 에이전트가 빈 섹션으로 예약 -->

## 6. 메모리 업데이트 제안 (선택)
<!-- 에이전트가 필요 시 자동 생성: memory.md에 추가할 후보 항목 -->
```

### 3.9 experiments_log.md 구조

```markdown
# 실험 이력

> 에이전트가 각 Cycle 완료 후 자동 업데이트.

| Cycle | 날짜 | 시작점 | 핵심 변경점 | 최종 avg reward (10ep) | 비고 |
|---|---|---|---|---|---|
| Cycle_1 | 2026-04-08 | base | 기본 구조 검증 | - | 초기 base 확인용 |
| Cycle_2 | - | base | - | - | - |
```

### 3.10 memory.md 구조

```
# 프로젝트 메모리

> 이 파일은 에이전트가 모든 작업 시작 전 반드시 읽어야 한다.
> 내용 추가/수정은 사용자 승인 후에만 가능하다. 거부된 제안은 재제안하지 않는다.

## 반복 실수 방지
- (예시) matplotlib figure 생성 시 subplot 사용 금지 — 반드시 개별 파일로 분리
- (예시) Cycle 폴더 내에서만 코드 수정, base/ 절대 건드리지 않기

## 환경/도구 관련
- Windows에서 MUJOCO_GL=egl 설정 시 MuJoCo 충돌 — 자동 제거 필요 (원본 코드에 이미 처리됨)

## 코드 패턴
- Gymnasium step()은 5-tuple 반환 — terminated, truncated 분리 처리 (step_env()에서 통합)
- DOB dhat은 에피소드 시작 시 zeros(2)로 리셋 (에피소드마다 리셋)
- ResidualDxNet과 NormalizedRBFModel은 rollout 중 동결; Phase 1(에피소드 시작 전)에서만 업데이트
- real_buffer와 model_buffer는 분리된 ReplayBufferDOB 인스턴스로 운영
- Checkpoint 파일명: Champion_Seed{run_idx}_BestModel.pt — 시드번호 포함으로 병렬 쓰기 충돌 없음

## 실험 교훈
- (Cycle별로 승인된 교훈이 누적)
```

### 3.11 config.py 스펙

```python
from dataclasses import dataclass, field

@dataclass
class DOBMBRLConfig:
    # 학습 기본
    num_episodes: int = 200
    max_steps_per_ep: int = 500
    warm_start_samples: int = 200

    # Q-Network
    lr_critic: float = 1e-3
    discount_factor: float = 0.99
    tau: float = 0.005              # soft-update coefficient
    update_interval: int = 10
    num_gradient_steps: int = 2

    # Exploration
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.005

    # Buffer
    buffer_size: int = int(1e5)
    mini_batch_size: int = 256
    num_epochs: int = 5             # residual model training epochs

    # Model
    real_ratio: float = 0.2         # sample_mixed_minibatch real 비율

    # DOB
    dob_w: float = 0.1              # DOB 가중치

    # RBF
    num_rbf_centers: int = 600
    rbf_width: float = 0.1
    rbf_initial_value: float = 5.0
    lr_rbf: float = 0.5
    lr_residual: float = 1e-2

    # Rollout
    max_horizon_length: int = 10
    uncertainty_threshold: float = 0.1
    num_generate_sample_iteration: int = 20
    epsilon_min_model: float = 0.1

    def __post_init__(self):
        assert self.mini_batch_size <= self.buffer_size, \
            f"mini_batch_size ({self.mini_batch_size}) must be <= buffer_size ({self.buffer_size})"
        assert 0.0 < self.real_ratio <= 1.0, \
            f"real_ratio must be in (0, 1], got {self.real_ratio}"
        assert 0.0 < self.tau <= 1.0, \
            f"tau must be in (0, 1], got {self.tau}"
        assert self.epsilon_min < self.epsilon, \
            f"epsilon_min must be < epsilon"
```

### 3.12 학습 로그 포맷

원본 코드 분석 결과 확인된 로그 데이터 구조:

#### 단일 시드 (`main.py`)
`train_DOB_core()` 반환값:
- `episode_cumulative_reward_vector`: `list[float]` — 에피소드별 누적 reward
- `episode_step_vector`: `list[int]` — 에피소드별 total_step_ct (누적 스텝 수)

저장 형식: `Cycle_N/logs/seed_{N}_result.pkl`
```python
{'rewards': episode_cumulative_reward_vector, 'steps': episode_step_vector}
```

#### 멀티 시드 (`run_multi_seed.py`)
저장 형식: `Cycle_N/results/DOB_MBRL_MultiSeed_Result.pkl`
```python
{'all_rewards': list[list[float]], 'all_steps': list[list[int]]}
```

#### result_queue (실시간 진행 모니터링)
```python
{'run_idx': int, 'ep_idx': int, 'reward': float, 'step': int}
```

**plot_metrics.py가 파싱할 키**: `rewards` (에피소드별), `steps` (총 스텝 축), `all_rewards` (멀티시드)

### 3.13 에이전트 구조

**단일 에이전트** 구조를 채택한다.

**근거**: 워크플로우가 순차적이고 도메인 지식이 하나(RL/DOB-MBRL)로 통일되어 있으며, 지시서를 7개 파일로 분리하여 필요 시에만 참조하므로 컨텍스트 윈도우 부담이 크지 않다. 서브에이전트 간 조율 오버헤드를 피한다.

### 3.14 Resume 로직 스펙

`main.py --resume` 인자 추가:
- checkpoint 파일(`Champion_Seed{N}_BestModel.pt`)이 존재하면 state_dict 로드
- 저장된 `episode`와 `total_steps`에서 학습 재개
- epsilon은 저장 시점의 값에서 이어서 decay

```python
# trainer.py에서 resume 처리
if resume and os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    q_network.load_state_dict(ckpt['q_network'])
    res_net.load_state_dict(ckpt['res_net'])
    uncert_model.load_state_dict(ckpt['uncert_model'])
    start_episode = ckpt['episode'] + 1
    total_step_ct = ckpt['total_steps']
```

### 3.15 requirements.txt (프로젝트 초기화 시 생성)

```
torch>=2.0.0
gymnasium>=0.29.0
gymnasium[classic-control]
numpy>=1.24.0
matplotlib>=3.7.0
```

### 3.16 확장성 설계

새로운 알고리즘, 환경, 모델을 추가할 때의 가이드이다. **모든 확장 작업은 반드시 `Cycle_N/` 내에서만 수행한다.** Base 프로젝트는 수정하지 않으며, 확장이 검증된 후 사용자가 명시적으로 Base 반영을 지시할 때만 Base를 업데이트한다.

> 아래 경로는 모두 `Cycle_N/` 기준이다.

#### 새 RL 알고리즘 추가

1. `Cycle_N/dob_mbrl/training/` 하위에 새 trainer 파일 추가 (예: `trainer_ppo.py`)
2. `Cycle_N/dob_mbrl/training/config.py`에 해당 알고리즘의 config 추가
3. `Cycle_N/main.py`에서 `--algo` 인자로 선택 가능하도록 분기

#### 새 환경 추가

1. `Cycle_N/dob_mbrl/envs/` 하위에 새 환경 유틸리티 파일 추가
2. `Cycle_N/dob_mbrl/dynamics/constants.py`에 해당 환경의 상수 추가 또는 별도 파일
3. `Cycle_N/main.py`에서 `--env` 인자로 선택

#### 새 네트워크 구조 추가

1. `Cycle_N/dob_mbrl/models/` 하위에 새 모델 파일 추가
2. trainer에서 config 기반으로 모델 인스턴스 선택

#### Base 반영 절차 (사용자 명시 지시 시에만)

Cycle에서 검증된 확장을 Base에 반영하고 싶을 때, 사용자가 "Base에 반영해줘"라고 명시적으로 지시하면:

1. 해당 Cycle의 변경 파일을 Base에 병합
2. Base의 임포트 테스트 재실행
3. CYCLE_REPORT.md에 "Base 반영 완료" 기록
4. **기존 Cycle에는 영향 없음** — 신규 Cycle부터 변경된 Base 적용

### 3.17 학습 실행 방식

코드 분석 결과, `Multi_Seed_DOB_Exp.py`는 `multiprocessing.Pool`로 병렬 학습을 수행하며, `train_DOB_core()`는 단일 시드 학습 함수이다. 재구성 시 **두 모드 모두 지원**한다:

- `main.py` — 단일 시드 학습 (디버깅, 빠른 검증용, resume 지원)
- `run_multi_seed.py` — 멀티 시드 병렬 학습 (정식 실험용)

**checkpoint 경로 주입**: `main.py`/`run_multi_seed.py`가 `--checkpoint-dir` 인자로 받아 `trainer.py`에 전달. trainer.py는 경로에 무관하게 동작.

**병렬 쓰기 충돌 방지**: 파일명에 시드 번호 포함 (`Champion_Seed{run_idx}_BestModel.pt`) — 동시 쓰기 충돌 없음.

---

## 부록: 검증 체크리스트

### 프로젝트 초기화 완료 기준

- [ ] 모든 디렉토리가 설계서 구조와 일치
- [ ] 원본 코드 2개 파일이 `base/original/`에 보관
- [ ] 모든 `__init__.py` 파일 존재
- [ ] `from dob_mbrl.models import QNetwork, ResidualDxNet, NormalizedRBFModel` 성공
- [ ] `from dob_mbrl.utils.buffer import ReplayBufferDOB` 성공
- [ ] `from dob_mbrl.training.trainer import train_DOB_core` 성공
- [ ] `python main.py --help` 정상 출력 (checkpoint-dir, seed, resume 인자 포함)
- [ ] `memory.md` 생성 완료
- [ ] `experiments_log.md` 생성 완료
- [ ] `requirements.txt` 생성 완료
- [ ] `.gitignore` 생성 완료 (checkpoints/, logs/, figures/, results/ 포함)
- [ ] CLAUDE.md + 7개 지시서 파일 생성 완료
- [ ] 스킬 파일 (`cycle-manager/SKILL.md`, `plot-generator/SKILL.md`) 생성 완료
- [ ] `config.py`의 `__post_init__` 검증 동작 확인

### Cycle 완료 기준

- [ ] `Cycle_N/` 디렉토리 존재
- [ ] 변경된 파일이 `Cycle_N/` 내에만 존재 (base 미수정)
- [ ] `Cycle_N/checkpoints/` 에 체크포인트 저장
- [ ] `Cycle_N/figures/` 에 개별 figure 파일 존재 (subplot 없음)
- [ ] `Cycle_N/CYCLE_REPORT.md` 작성 완료 (필수 섹션 충족, 섹션 5는 빈 예약)
- [ ] `experiments_log.md`에 이번 Cycle 항목 추가 완료
- [ ] 메모리 업데이트 제안 여부 확인
