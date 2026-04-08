# 프로젝트 메모리

> 이 파일은 에이전트가 모든 작업 시작 전 반드시 읽어야 한다.
> 내용 추가/수정은 사용자 승인 후에만 가능하다. 거부된 제안은 재제안하지 않는다.

## 반복 실수 방지

- matplotlib figure 생성 시 subplot 사용 금지 — 반드시 개별 파일로 분리
- Cycle 폴더 내에서만 코드 수정, `base/` 절대 건드리지 않기
- `config.py` 수정은 파일 직접 편집만. CLI override 없음

## 환경/도구 관련

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
