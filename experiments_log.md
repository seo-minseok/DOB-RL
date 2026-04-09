# 실험 이력

> 에이전트가 각 Cycle 완료 후 자동 업데이트.
> 기존 행은 수정하지 않음 — 항상 새 행 추가.

| Cycle | 날짜 | 시작점 | 핵심 변경점 | 최종 avg reward (10ep) | 비고 |
|---|---|---|---|---|---|
| base | 2026-04-08 | — | 초기 구조 설정 (모듈화) | — | 프로젝트 초기화 |
| Cycle_1 | 2026-04-09 | base | Ablation: uncertainty-weighted vs uniform sampling, 저장 경로 정비, 비교 시각화 추가 | Baseline 280.59 ± 162.29 / Ablation 323.65 ± 172.29 (16 seeds, 200 ep) | 두 조건 모두 2/16 seeds만 ≥480 유지 — 안정성이 핵심 병목 |
