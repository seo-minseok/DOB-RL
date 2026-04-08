# Skill: plot-generator

학습 로그 파일에서 metric별 개별 figure를 생성한다. 1 figure per file 규칙을 자동 검증한다.

## 트리거 조건

사용자가 다음을 요청할 때:
- "figure 그려줘"
- "결과 시각화해줘"
- "학습 결과 플롯해줘"

## 실행 방법

```bash
# 단일 시드 결과
python Cycle_N/scripts/plot_results.py --log-dir Cycle_N/logs --seed 1 --figures-dir Cycle_N/figures

# 멀티 시드 결과
python Cycle_N/scripts/plot_results.py --results-dir Cycle_N/results --multi-seed --figures-dir Cycle_N/figures
```

## 출력

`Cycle_N/figures/` 하위에 metric별 개별 PNG 파일:
- `reward_curve.png`
- `reward_smoothed.png`
- `total_steps.png`
- `multiseed_mean_std.png` (멀티시드)

## 핵심 규칙

- `plt.subplots()` 호출 시 반드시 단일 Axes `(1, 1)`.
- `save_figure(fig, path)` 함수가 Axes 수를 검증 — AssertionError 시 subplot 분리 필요.

## 스크립트 위치

`Cycle_N/scripts/plot_results.py` (base에서 복사됨)
