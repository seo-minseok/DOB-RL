# 03 — Cycle 생성/관리 절차

## Cycle 생성 절차

1. **에이전트 구두 확인**: "base에서 시작할까요, 아니면 특정 Cycle(예: Cycle_2)에서 시작할까요?"
2. **사용자 확인 후** `create_cycle.sh` 실행:
   ```bash
   bash .claude/skills/cycle-manager/scripts/create_cycle.sh --from base
   # 또는
   bash .claude/skills/cycle-manager/scripts/create_cycle.sh --from Cycle_2
   ```
3. Cycle 번호는 기존 Cycle 중 최대 번호 + 1 자동 부여.

## Cycle 시작점 정책

- 기본값: base에서 복사
- 사용자가 "Cycle_N에서 시작" 명시 시: 해당 Cycle의 코드만 복사 (checkpoints/logs/figures/results는 복사 안 함)
- 실패한 Cycle 코드 계승: 사용자가 명시적으로 지정 ("Cycle_2 코드 기반으로 Cycle_5 만들어줘")

## Cycle 폴더 구조

```
Cycle_N/
├── CYCLE_REPORT.md
├── dob_mbrl/          # (소스에서 복사)
├── main.py
├── run_multi_seed.py
├── scripts/
├── checkpoints/       # .gitignore (빈 폴더로 생성)
├── figures/           # .gitignore (빈 폴더로 생성)
├── logs/              # .gitignore (빈 폴더로 생성)
└── results/           # .gitignore (빈 폴더로 생성)
```

## CYCLE_REPORT.md 필수 섹션

```markdown
# Cycle N 실험 보고서

> 생성일: YYYY-MM-DD
> 시작점: base / Cycle_N

## 1. 변경점 요약
## 2. 변경 상세
## 3. 하이퍼파라미터
## 4. 학습 결과
## 5. 관찰 및 교훈
<!-- ★ 사람이 직접 작성 — 에이전트가 빈 섹션으로 예약 -->
## 6. 메모리 업데이트 제안 (선택)
```

## Base 반영 절차 (사용자 명시 지시 시에만)

1. Cycle의 변경 파일을 base에 병합
2. base 임포트 테스트 재실행
3. CYCLE_REPORT.md에 "Base 반영 완료" 기록
4. 기존 Cycle 영향 없음 — 신규 Cycle부터 변경된 base 적용
