# 01 — 프로젝트 전반 규칙

## Base 보호 절대 규칙

- `base/` 디렉토리와 그 하위 모든 파일은 **사용자가 명시적으로 "base 수정해줘"라고 지시하지 않는 한 절대 수정 금지.**
- `base/original/train_DOB_core.py`, `base/original/Multi_Seed_DOB_Exp.py`는 어떤 경우에도 수정 금지. 참조용 읽기만 허용.
- 에이전트가 Base 수정이 필요하다고 판단되면 → **에스컬레이션**: 사용자에게 허가를 명시적으로 요청.

## Cycle 내 작업 원칙

- 모든 코드 변경은 `cycles/Cycle_N/` 내에서만 수행.
- Base를 수정하지 않고 Cycle 범위 내에서 해결할 수 없는 경우에만 Base 수정 에스컬레이션 허용.

## Base 수정 전파 정책 (사용자 승인 시)

1. 에이전트가 Base 파일 수정 수행
2. Base 임포트 테스트 재실행: `python -c "from dob_mbrl.models import QNetwork"`
3. **기존 cycles/ 내 Cycle은 영향 없음** — 이미 완료된 Cycle은 수정 전 Base 기반
4. **신규 Cycle부터 새 Base 적용**

## 원칙 위반 판단 기준

에이전트가 다음을 하려 할 때는 반드시 멈추고 사용자에게 확인:
- `base/` 하위 파일 Edit/Write
- `base/original/` 하위 파일 접근 (읽기 제외)
- 특정 `cycles/Cycle_N/` 범위를 벗어나 다른 Cycle 파일 수정
