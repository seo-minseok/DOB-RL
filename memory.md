# 프로젝트 메모리

> 이 파일은 에이전트가 모든 작업 시작 전 반드시 읽어야 한다.
> 내용 추가/수정은 사용자 승인 후에만 가능하다. 거부된 제안은 재제안하지 않는다.

## 반복 실수 방지

- matplotlib figure 생성 시 subplot 사용 금지 — 반드시 개별 파일로 분리
- Cycle 폴더 내에서만 코드 수정, `base/` 절대 건드리지 않기
- `config.py` 수정은 파일 직접 편집만. CLI override 없음
- **학습 프로세스 중복 실행 절대 금지 (반복 위반 주의)**:
  - 학습 실행 직후 프로세스 목록에 바로 안 보이거나, 확인 명령이 빈 결과를 반환해도 **절대 재실행하지 말 것**.
  - 프로세스 목록은 타이밍 문제로 비어 보일 수 있음. **CSV 파일에 새 행이 추가되고 있으면 학습은 정상 동작 중**이다.
  - **warm_start 구간(기본 10,000 step)에서는 CSV가 아직 생성되지 않는다** — CSV 없음이 곧 미실행 증거가 아님.
  - 실행 여부 판단 기준(우선순위 순):
    1. **`results/*/config.json` 생성 여부** — 학습 진입 즉시 생성됨. 존재하면 이미 실행 중이거나 실행됐던 것.
    2. **CSV 파일 행 수 증가 여부** — warm_start 이후 생성. 없다고 미실행이 아님.
    3. `Get-CimInstance Win32_Process | Where-Object { $_.Name -like "*python*" } | Select-Object ProcessId, CommandLine` — 커맨드라인까지 확인해야 어떤 시드인지 식별 가능.
  - 하나라도 "실행 중"을 가리키면 **추가 실행 금지**.
  - **`python main.py`를 "테스트" 또는 "확인" 목적으로 실행하는 것도 중복 실행이다** — `| head -5` 등 파이프를 붙여도 학습 프로세스 자체는 백그라운드에서 계속 실행된다. 절대 금지.
  - 프로세스가 안 보이면 **30초 이상 대기 후 재확인**이 원칙. 재실행이 아님.
  - 같은 CSV에 두 프로세스가 동시에 쓰면 데이터가 영구 오염됨.

- **학습 실행 시 반드시 백그라운드로 실행** (cmd 창 열기 금지):
  - `Start-Process ... -WindowStyle Normal` 절대 사용 금지 — cmd 창이 열리고, 창 닫으면 프로세스도 종료됨
  - 반드시 Bash 툴의 `run_in_background: true` 사용, 출력은 로그 파일로 리디렉션
  - 예: `python main.py ... > logs/seed1.log 2>&1`

## 환경/도구 관련

- **학습 실행은 반드시 `dob-rl` conda 환경 + tmux 세션 안에서**: `tmux new -s <session_name>`으로 세션 생성 → `conda activate dob-rl` → 학습 실행. 터미널이 닫혀도 프로세스가 유지됨. PyTorch(cu128, Blackwell sm_120 지원), gymnasium[mujoco], numpy 등 모든 의존성이 이 환경에 설치돼 있음.
- Windows에서 `MUJOCO_GL=egl` 설정 시 MuJoCo 충돌 — 자동 제거 필요 (원본 코드에 이미 처리됨)
- Gymnasium `step()`은 5-tuple 반환 — `terminated`, `truncated` 분리 처리 (`step_env()`에서 통합)

## 코드 패턴

- DOB `dhat`은 에피소드 시작 시 `zeros(2)`로 리셋 (에피소드마다 리셋)
- `ResidualDxNet`과 `NormalizedRBFModel`은 rollout 중 동결 (Phase 1 에피소드 시작 전에만 업데이트)
- `real_buffer`와 `model_buffer`는 분리된 `ReplayBufferDOB` 인스턴스로 운영
- Checkpoint 파일명: `Champion_Seed{run_idx}_BestModel.pt` — 시드 번호 포함으로 병렬 쓰기 충돌 없음
- RBF rollout 첫 번째 step (`h==0`)에서는 항상 신뢰 가능으로 처리 (`is_reliable[:] = True`)

## 실험 교훈

- (Cycle별로 승인된 교훈이 누적됨)
