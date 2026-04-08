# 05 — Figure 생성 규칙

## 핵심 규칙: 1 figure per file

- **하나의 이미지 파일 = 하나의 figure.**
- `plt.subplots()` 호출 시 반드시 `(1, 1)` 또는 단일 Axes.
- subplot 배열(`(2, 2)`, `(1, 3)` 등) **금지.**
- 위반 시 `save_figure()` 함수가 AssertionError 발생.

## 예외: Cycle 간 비교 오버레이

서로 다른 설정의 mean±std를 **하나의 Axes에 겹쳐 그리는 것은 허용**.
이는 subplot이 아니라 단일 figure 내 오버레이이므로 규칙에 위배되지 않음.

```python
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(steps_A, mean_A, label='Cycle_1')
ax.plot(steps_B, mean_B, label='Cycle_2')
ax.fill_between(steps_A, lower_A, upper_A, alpha=0.2)
ax.fill_between(steps_B, lower_B, upper_B, alpha=0.2)
```

## 파일 명명 규칙

| metric | 파일명 |
|---|---|
| 에피소드 reward | `reward_curve.png` |
| 스무딩 reward | `reward_smoothed.png` |
| 총 스텝 | `total_steps.png` |
| 멀티시드 mean±std | `multiseed_mean_std.png` |
| Cycle 간 비교 | `compare_cycle{A}_vs_cycle{B}.png` |

## Figure 저장 경로

`Cycle_N/figures/` 하위에 저장.

## 플롯 스크립트 실행

```bash
# 단일 시드
python scripts/plot_results.py --log-dir ./logs --seed 1

# 멀티 시드
python scripts/plot_results.py --results-dir ./results --multi-seed
```

## save_figure() 함수 반드시 사용

`scripts/plot_results.py`의 `save_figure(fig, path)` 함수는 Axes 수를 검증함.
직접 `fig.savefig()` 호출 시 subplot 검증이 누락될 수 있으므로 반드시 `save_figure()` 사용.
