#!/usr/bin/env bash
# create_cycle.sh — 새 Cycle 폴더 생성
# Usage:
#   bash create_cycle.sh --from base
#   bash create_cycle.sh --from Cycle_2
#
# 반드시 프로젝트 루트에서 실행할 것.

set -euo pipefail

FROM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --from) FROM="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$FROM" ]]; then
    echo "Usage: bash create_cycle.sh --from base|Cycle_N"
    exit 1
fi

# 소스 경로 결정
if [[ "$FROM" == "base" ]]; then
    SRC="base"
else
    SRC="$FROM"
fi

if [[ ! -d "$SRC" ]]; then
    echo "Error: source directory '$SRC' not found."
    exit 1
fi

# Cycle 루트 디렉토리
CYCLES_ROOT="cycles"
mkdir -p "$CYCLES_ROOT"

# 다음 Cycle 번호 결정
MAX_N=0
for d in "$CYCLES_ROOT"/Cycle_*/; do
    if [[ -d "$d" ]]; then
        N="${d//[^0-9]/}"
        if [[ -n "$N" && "$N" -gt "$MAX_N" ]]; then
            MAX_N=$N
        fi
    fi
done
NEXT_N=$((MAX_N + 1))
DEST="$CYCLES_ROOT/Cycle_${NEXT_N}"

echo "Creating $DEST from $SRC ..."

# 코드만 복사 (checkpoints/logs/figures/results 제외)
mkdir -p "$DEST"
rsync -a \
    --exclude='checkpoints/' \
    --exclude='logs/' \
    --exclude='figures/' \
    --exclude='results/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='CYCLE_REPORT.md' \
    "$SRC/" "$DEST/"

# 빈 출력 디렉토리 생성
mkdir -p "$DEST/checkpoints" "$DEST/figures" "$DEST/logs" "$DEST/results"

# CYCLE_REPORT.md 템플릿 생성
TODAY=$(date +%Y-%m-%d)
cat > "$DEST/CYCLE_REPORT.md" <<EOF
# Cycle ${NEXT_N} 실험 보고서

> 생성일: ${TODAY}
> 시작점: ${FROM}

## 1. 변경점 요약
<!-- 에이전트 자동 생성: 이 Cycle에서 무엇을 변경했는가 -->

## 2. 변경 상세
<!-- 에이전트 자동 생성: 수정된 파일 목록 및 각 파일의 변경 내용 -->

## 3. 하이퍼파라미터
<!-- 에이전트 자동 생성: 이 Cycle에서 사용된 주요 하이퍼파라미터 값 -->

## 4. 학습 결과
<!-- 에이전트 자동 생성: 정량 지표 + figure 파일 경로 목록 -->

## 5. 관찰 및 교훈
<!-- ★ 사람이 직접 작성 — 에이전트가 빈 섹션으로 예약 -->

## 6. 메모리 업데이트 제안 (선택)
<!-- 에이전트가 필요 시 자동 생성 -->
EOF

echo "Done: $DEST created from $SRC"
echo "  Code:    $DEST/dob_mbrl/, $DEST/main.py, $DEST/run_multi_seed.py"
echo "  Empty:   $DEST/checkpoints/ $DEST/figures/ $DEST/logs/ $DEST/results/"
echo "  Report:  $DEST/CYCLE_REPORT.md"
