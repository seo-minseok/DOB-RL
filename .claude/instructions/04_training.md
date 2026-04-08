# 04 — 학습 실행 지침

## CLI 스펙

```bash
# 단일 시드 학습
python main.py --checkpoint-dir ./checkpoints --seed 1

# Resume (마지막 체크포인트에서 재개)
python main.py --checkpoint-dir ./checkpoints --seed 1 --resume

# 멀티 시드 병렬 학습
python run_multi_seed.py --checkpoint-dir ./checkpoints --num-seeds 16
```

**학습 실행은 사용자가 직접 수행.** 에이전트는 정확한 명령만 안내.

## Checkpoint 저장

- 경로: `cycles/Cycle_N/checkpoints/Champion_Seed{seed}_BestModel.pt`
- 조건: 최근 10 에피소드 평균 reward ≥ 480 AND 기존 최고 초과 시 저장
- 파일명에 시드 번호 포함 → 병렬 쓰기 충돌 없음

## Resume 로직

`main.py --resume` 시:
- `checkpoint_path`가 존재하면 state_dict 로드 (q_network, res_net, uncert_model)
- `start_episode = ckpt['episode'] + 1`에서 재개
- epsilon은 저장 시점 값에서 이어서 decay

## 로그 저장 형식

단일 시드: `cycles/Cycle_N/logs/seed_{N}_result.pkl`
```python
{'rewards': list[float], 'steps': list[int]}
```

멀티 시드: `cycles/Cycle_N/results/DOB_MBRL_MultiSeed_Result.pkl`
```python
{'all_rewards': list[list[float]], 'all_steps': list[list[int]]}
```

## 하이퍼파라미터 수정

`cycles/Cycle_N/dob_mbrl/training/config.py`를 직접 편집. CLI override 없음.
수정 후 반드시 `__post_init__` 검증이 통과하는지 확인:
```bash
python -c "from dob_mbrl.training.config import DOBMBRLConfig; DOBMBRLConfig()"
```
