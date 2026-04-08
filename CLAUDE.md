# DOB-MBRL 프로젝트 에이전트 지침

DOB-MBRL(Disturbance Observer-Based Model-Based RL) — CartPole에서 nominal dynamics + learned residual + DOB 온라인 보정을 결합한 강화학습 프로젝트.

---

## 필수 선행 작업

**모든 작업 시작 전 반드시 `memory.md`를 읽어라.** 반복 실수, 코드 패턴, 실험 교훈이 기록되어 있다.

---

## 절대 규칙 (인라인)

1. **`base/` 디렉토리는 절대 수정 금지.** 사용자가 명시적으로 "base 수정해줘"라고 지시할 때만 허용.
2. **`base/original/` 내 원본 파일(`train_DOB_core.py`, `Multi_Seed_DOB_Exp.py`)은 절대 수정 금지.**
3. **모든 코드 변경은 `Cycle_N/` 내에서만 수행.**
4. **하나의 이미지 파일 = 하나의 figure.** `plt.subplots()` 호출 시 반드시 단일 Axes `(1,1)`. subplot 금지.
   - 예외: Cycle 간 비교용 mean±std 오버레이는 단일 Axes에 겹쳐 그리는 것이므로 허용.
5. **`config.py` 수정은 파일 직접 편집만.** CLI override 없음.

---

## 조건부 지시서 로딩 규칙

작업 유형에 따라 해당 지시서를 읽어라:

| 작업 유형 | 읽을 지시서 |
|---|---|
| Base 구조 수정 / 임포트 관련 | `.claude/instructions/02_code-structure.md` |
| Cycle 생성 요청 | `.claude/instructions/03_cycle-management.md` |
| 학습 실행 / checkpoint / resume | `.claude/instructions/04_training.md` |
| Figure 생성 | `.claude/instructions/05_plotting.md` |
| 보고서 작성 | `.claude/instructions/06_reporting.md` |
| memory.md 수정 | `.claude/instructions/07_memory-protocol.md` |
| Base 보호 규칙 확인 | `.claude/instructions/01_project-rules.md` |

---

## 지시서 목록

| 파일 | 역할 |
|---|---|
| `01_project-rules.md` | Base 보호, Cycle 내 작업 원칙, 원본 파일 수정 금지 |
| `02_code-structure.md` | 패키지 구조, 모듈 임포트 규칙, 새 모듈 추가 위치 |
| `03_cycle-management.md` | Cycle 생성 절차, 번호 부여, 복사 범위, CYCLE_REPORT 템플릿 |
| `04_training.md` | 학습 CLI 스펙, checkpoint 경로, resume 로직, 로그 포맷 |
| `05_plotting.md` | 1 figure per file 규칙, 오버레이 허용 조건, 파일 명명 |
| `06_reporting.md` | CYCLE_REPORT.md 필수 섹션, experiments_log.md 업데이트 |
| `07_memory-protocol.md` | memory.md 읽기/쓰기 프로토콜, 승인 흐름 |

---

## 스킬

- **cycle-manager**: Cycle 폴더 생성 → `.claude/skills/cycle-manager/SKILL.md`
- **plot-generator**: 로그 → figure 생성 → `.claude/skills/plot-generator/SKILL.md`
