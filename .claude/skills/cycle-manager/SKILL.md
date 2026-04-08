# Skill: cycle-manager

새 실험 Cycle 폴더를 생성한다. Base 또는 지정 Cycle에서 코드를 복사하고, 빈 출력 디렉토리를 초기화한다.

## 트리거 조건

사용자가 다음을 요청할 때:
- "새 Cycle 시작해줘"
- "Cycle 만들어줘"
- "Cycle_N에서 새 Cycle 만들어줘"

## 실행 전 에이전트 확인 사항

스크립트 실행 전 반드시 사용자에게 구두 확인:
> "base에서 시작할까요, 아니면 특정 Cycle(예: Cycle_2)에서 시작할까요?"

## 실행 방법

```bash
bash .claude/skills/cycle-manager/scripts/create_cycle.sh --from base
# 또는
bash .claude/skills/cycle-manager/scripts/create_cycle.sh --from Cycle_2
```

## 출력

- `Cycle_N/` 디렉토리 (소스 코드 복사본)
- 빈 폴더: `checkpoints/`, `figures/`, `logs/`, `results/`
- `Cycle_N/CYCLE_REPORT.md` (템플릿)

## 검증

```bash
ls Cycle_N/dob_mbrl/
python -c "import sys; sys.path.insert(0,'Cycle_N'); from dob_mbrl.models import QNetwork"
```
