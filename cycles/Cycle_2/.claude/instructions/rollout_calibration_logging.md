# RBF Calibration + Rollout 품질 지표 로깅 구현 지시서

## 목적

다음 세 가지 지표를 추가한다:

| 지표 | 의미 | 추가 위치 |
|---|---|---|
| `rollout_pass_rate` | h>0 스텝에서 uncertainty threshold를 통과한 샘플 비율 | `rollout.py` |
| `rollout_avg_horizon` | 각 trajectory가 실제로 완료한 평균 스텝 수 | `rollout.py` |
| `rbf_calib_ratio` | RBF 예측 uncertainty 평균 / 실제 uncertainty 평균 | `model_learning.py`, `trainer.py` |
| `rbf_calib_corr` | RBF 예측 mag과 실제 mag의 Pearson correlation | `model_learning.py`, `trainer.py` |

수정 대상 파일은 3개다:
1. `cycles/Cycle_2/dob_mbrl/training/rollout.py`
2. `cycles/Cycle_2/dob_mbrl/training/model_learning.py`
3. `cycles/Cycle_2/dob_mbrl/training/trainer.py`

---

## 파일 1: `rollout.py`

### 변경 요약
- `generate_samples_dob` 내부에 `pass_rate_vals`, `horizon_counts` 수집 로직 추가
- 반환값을 3개 → 5개로 확장: `(model_buffer, rollout_uncert_avg, rollout_pass_rate, rollout_avg_horizon)`

### 전체 교체

기존 `generate_samples_dob` 함수 전체를 아래로 교체한다. (`sample_mixed_minibatch`는 건드리지 않는다.)

```python
def generate_samples_dob(real_buffer, model_buffer,
                          res_net, uncert_model, actor,
                          rollout_noise: float, options: dict,
                          p_nom: dict, use_nominal: bool):
    """
    Model rollout with per-step uncertainty gating.
    h==0 (첫 번째 step)에서는 항상 신뢰 가능으로 처리.
    ResidualDxNet과 NormalizedRBFModel은 이 함수 안에서 동결 (no_grad).

    actor        : ActorNetwork (연속 행동 출력)
    rollout_noise: 탐색 노이즈 std (config.epsilon_min_model 재사용)

    Returns
    -------
    model_buffer       : 갱신된 model_buffer
    rollout_uncert_avg : 롤아웃 전 스텝(h=0 포함)의 RBF 예측 uncertainty 평균
    rollout_pass_rate  : h>0 스텝에서 threshold를 통과한 샘플 비율 (0~1)
    rollout_avg_horizon: trajectory당 평균 완료 스텝 수 (max=max_horizon_length)
    """
    max_horizon      = options['max_horizon_length']
    uncert_threshold = options['uncertainty_threshold']
    num_iter         = options['num_generate_sample_iteration']
    B                = options['mini_batch_size']
    noise_std        = max(rollout_noise, options['epsilon_min_model'])

    uncert_mag_all = []
    pass_rate_vals = []   # h>0 스텝에서만 수집
    horizon_counts = []   # iteration마다 trajectory별 완료 step 수

    for _ in range(num_iter):
        if real_buffer.length < B:
            break

        idx         = np.random.randint(0, real_buffer.length, size=B)
        current_obs = torch.tensor(real_buffer.obs[idx])   # (B, 24)
        alive_mask  = np.ones(B, dtype=bool)
        traj_horizon = np.zeros(B, dtype=np.float32)       # 각 trajectory의 완료 스텝

        for h in range(max_horizon):
            n_alive = alive_mask.sum()
            if n_alive == 0:
                break

            valid_obs = current_obs[alive_mask]   # (n_alive, 24)

            with torch.no_grad():
                valid_act_t = actor(valid_obs)    # (n_alive, 4), tanh ∈ [-1, 1]
            valid_act_np = valid_act_t.numpy()

            # 탐색 노이즈 추가 후 클리핑
            noise    = np.random.normal(0.0, noise_std, size=valid_act_np.shape).astype(np.float32)
            valid_act = np.clip(valid_act_np + noise, -1.0, 1.0)

            valid_obs_np = valid_obs.cpu().numpy()
            inp_rbf = torch.tensor(
                np.concatenate([valid_obs_np, valid_act], axis=-1))
            with torch.no_grad():
                pred_uncert  = uncert_model(inp_rbf).cpu().numpy()  # (n_alive, 7)
            uncert_mag_arr   = np.linalg.norm(pred_uncert, axis=1)  # (n_alive,)
            uncert_mag_all.append(uncert_mag_arr)
            is_reliable      = uncert_mag_arr < uncert_threshold

            if h == 0:
                is_reliable[:] = True
            else:
                pass_rate_vals.append(float(is_reliable.mean()))

            n_reliable = is_reliable.sum()
            if n_reliable == 0:
                break

            rel_obs  = valid_obs_np[is_reliable]
            rel_act  = valid_act[is_reliable]
            rel_next = predict_next_obs_dob(rel_obs, rel_act, res_net, p_nom, use_nominal)
            rel_rew, rel_done = reward_is_done_function(rel_next)

            model_buffer.store_batch(rel_obs, rel_act, rel_next, rel_rew, rel_done)

            alive_idx    = np.where(alive_mask)[0]
            alive_mask[alive_idx[~is_reliable]] = False

            reliable_idx = alive_idx[is_reliable]
            traj_horizon[reliable_idx] += 1.0
            alive_mask[reliable_idx[rel_done]] = False

            not_done = ~rel_done
            if not_done.any():
                to_update = reliable_idx[not_done]
                current_obs[to_update] = torch.tensor(rel_next[not_done])

        horizon_counts.extend(traj_horizon.tolist())

    rollout_uncert_avg  = float(np.concatenate(uncert_mag_all).mean()) if uncert_mag_all  else float('nan')
    rollout_pass_rate   = float(np.mean(pass_rate_vals))                if pass_rate_vals  else float('nan')
    rollout_avg_horizon = float(np.mean(horizon_counts))                if horizon_counts  else float('nan')
    return model_buffer, rollout_uncert_avg, rollout_pass_rate, rollout_avg_horizon
```

---

## 파일 2: `model_learning.py`

### 변경 요약
- 파일 맨 끝에 `evaluate_rbf_calibration` 함수를 추가한다.
- 기존 코드는 한 글자도 수정하지 않는다.

### 추가할 코드 (파일 끝 `return loss_sum / max(1, ct)` 이후에 붙여넣기)

```python


def evaluate_rbf_calibration(uncert_model, real_buffer,
                              sample_size: int = 4096):
    """
    real_buffer 샘플에서 RBF 예측 uncertainty와 실제 uncertainty를 비교한다.

    Returns
    -------
    calib_ratio : mean(||RBF pred||₂) / mean(||actual||₂)
                  1.0이면 스케일 일치, <1이면 under-predict, >1이면 over-predict
    calib_corr  : Pearson correlation(pred_mag, actual_mag)
                  1.0에 가까울수록 RBF가 고-uncertainty 상태를 올바르게 식별
    """
    valid_len = real_buffer.length
    if valid_len == 0:
        return float('nan'), float('nan')

    n   = min(valid_len, sample_size)
    idx = np.random.choice(valid_len, size=n, replace=False)

    obs           = real_buffer.obs[idx]
    act           = real_buffer.act[idx]
    actual_uncert = real_buffer.uncertainty[idx]   # (n, 7)

    uncert_model.eval()
    with torch.no_grad():
        inp  = torch.tensor(np.concatenate([obs, act], axis=-1))
        pred = uncert_model(inp).cpu().numpy()     # (n, 7)

    pred_mag   = np.linalg.norm(pred,           axis=1)   # (n,)
    actual_mag = np.linalg.norm(actual_uncert,  axis=1)   # (n,)

    calib_ratio = float(pred_mag.mean() / (actual_mag.mean() + 1e-8))

    if pred_mag.std() < 1e-8 or actual_mag.std() < 1e-8:
        calib_corr = float('nan')
    else:
        calib_corr = float(np.corrcoef(pred_mag, actual_mag)[0, 1])

    return calib_ratio, calib_corr
```

---

## 파일 3: `trainer.py`

총 5곳을 수정한다.

---

### 수정 1: import 라인 (파일 상단 14번째 줄)

**기존:**
```python
from .model_learning import train_residual_dx_model_dob, train_uncertainty_rbf
```

**변경 후:**
```python
from .model_learning import train_residual_dx_model_dob, train_uncertainty_rbf, evaluate_rbf_calibration
```

---

### 수정 2: 에피소드 지표 초기값 선언 (기존 134~138번째 줄 블록)

**기존:**
```python
        ep_res_net_loss        = float('nan')
        ep_rbf_loss            = float('nan')
        ep_buffer_uncert_avg   = float('nan')
        ep_sampled_uncert_avg  = float('nan')
        ep_rollout_uncert_avg  = float('nan')
```

**변경 후:**
```python
        ep_res_net_loss        = float('nan')
        ep_rbf_loss            = float('nan')
        ep_buffer_uncert_avg   = float('nan')
        ep_sampled_uncert_avg  = float('nan')
        ep_rollout_uncert_avg  = float('nan')
        ep_rollout_pass_rate   = float('nan')
        ep_rollout_avg_horizon = float('nan')
        ep_rbf_calib_ratio     = float('nan')
        ep_rbf_calib_corr      = float('nan')
```

---

### 수정 3: `train_uncertainty_rbf` 호출 직후 (기존 151~154번째 줄)

**기존:**
```python
                ep_rbf_loss = train_uncertainty_rbf(
                    uncert_model, rbf_opt, real_buffer, res_net,
                    cfg.mini_batch_size, 5
                )
```

**변경 후:**
```python
                ep_rbf_loss = train_uncertainty_rbf(
                    uncert_model, rbf_opt, real_buffer, res_net,
                    cfg.mini_batch_size, 5
                )
                ep_rbf_calib_ratio, ep_rbf_calib_corr = evaluate_rbf_calibration(
                    uncert_model, real_buffer
                )
```

---

### 수정 4: `generate_samples_dob` 호출 및 반환값 언패킹 (기존 156~160번째 줄)

**기존:**
```python
                model_buffer, ep_rollout_uncert_avg = generate_samples_dob(
                    real_buffer, model_buffer, res_net, uncert_model,
                    actor, cfg.epsilon_min_model, sample_gen_options,
                    p_nom, use_nominal
                )
```

**변경 후:**
```python
                (model_buffer,
                 ep_rollout_uncert_avg,
                 ep_rollout_pass_rate,
                 ep_rollout_avg_horizon) = generate_samples_dob(
                    real_buffer, model_buffer, res_net, uncert_model,
                    actor, cfg.epsilon_min_model, sample_gen_options,
                    p_nom, use_nominal
                )
```

---

### 수정 5: `ep_metrics` dict + CSV 헤더/행 (기존 303~344번째 줄)

**기존 `ep_metrics` dict:**
```python
        ep_metrics = {
            'nominal_error_avg':   float(np.mean(ep_nominal_errors))   if ep_nominal_errors   else float('nan'),
            'residual_error_avg':  float(np.mean(ep_residual_errors))  if ep_residual_errors  else float('nan'),
            'dhat_norm_avg':       float(np.mean(ep_dhat_norms))       if ep_dhat_norms       else float('nan'),
            'uncertainty_avg':     float(np.mean(ep_uncertainty_mags)) if ep_uncertainty_mags else float('nan'),
            'res_net_loss':        ep_res_net_loss,
            'rbf_loss':            ep_rbf_loss,
            'td_loss_avg':         float(np.mean(ep_td_losses)) if ep_td_losses else float('nan'),
            'episode_length':      step_ct,
            'expl_noise':          cfg.expl_noise,
            'buffer_uncert_avg':    ep_buffer_uncert_avg,
            'sampled_uncert_avg':   ep_sampled_uncert_avg,
            'rollout_uncert_avg':   ep_rollout_uncert_avg,
        }
```

**변경 후:**
```python
        ep_metrics = {
            'nominal_error_avg':    float(np.mean(ep_nominal_errors))   if ep_nominal_errors   else float('nan'),
            'residual_error_avg':   float(np.mean(ep_residual_errors))  if ep_residual_errors  else float('nan'),
            'dhat_norm_avg':        float(np.mean(ep_dhat_norms))       if ep_dhat_norms       else float('nan'),
            'uncertainty_avg':      float(np.mean(ep_uncertainty_mags)) if ep_uncertainty_mags else float('nan'),
            'res_net_loss':         ep_res_net_loss,
            'rbf_loss':             ep_rbf_loss,
            'td_loss_avg':          float(np.mean(ep_td_losses)) if ep_td_losses else float('nan'),
            'episode_length':       step_ct,
            'expl_noise':           cfg.expl_noise,
            'buffer_uncert_avg':    ep_buffer_uncert_avg,
            'sampled_uncert_avg':   ep_sampled_uncert_avg,
            'rollout_uncert_avg':   ep_rollout_uncert_avg,
            'rollout_pass_rate':    ep_rollout_pass_rate,
            'rollout_avg_horizon':  ep_rollout_avg_horizon,
            'rbf_calib_ratio':      ep_rbf_calib_ratio,
            'rbf_calib_corr':       ep_rbf_calib_corr,
        }
```

**기존 CSV 헤더 `writer.writerow` (기존 329~335번째 줄):**
```python
                if write_header:
                    writer.writerow([
                        'seed', 'episode', 'total_steps', 'reward',
                        'nominal_error_avg', 'residual_error_avg', 'dhat_norm_avg',
                        'uncertainty_avg', 'res_net_loss', 'rbf_loss', 'td_loss_avg',
                        'episode_length', 'expl_noise', 'buffer_uncert_avg',
                        'sampled_uncert_avg', 'rollout_uncert_avg',
                    ])
```

**변경 후:**
```python
                if write_header:
                    writer.writerow([
                        'seed', 'episode', 'total_steps', 'reward',
                        'nominal_error_avg', 'residual_error_avg', 'dhat_norm_avg',
                        'uncertainty_avg', 'res_net_loss', 'rbf_loss', 'td_loss_avg',
                        'episode_length', 'expl_noise', 'buffer_uncert_avg',
                        'sampled_uncert_avg', 'rollout_uncert_avg',
                        'rollout_pass_rate', 'rollout_avg_horizon',
                        'rbf_calib_ratio', 'rbf_calib_corr',
                    ])
```

**기존 CSV 데이터 행 `writer.writerow` (기존 336~344번째 줄):**
```python
                writer.writerow([
                    run_idx, episode_ct, total_step_ct, episode_reward,
                    ep_metrics['nominal_error_avg'], ep_metrics['residual_error_avg'],
                    ep_metrics['dhat_norm_avg'], ep_metrics['uncertainty_avg'],
                    ep_metrics['res_net_loss'], ep_metrics['rbf_loss'],
                    ep_metrics['td_loss_avg'], ep_metrics['episode_length'],
                    ep_metrics['expl_noise'], ep_metrics['buffer_uncert_avg'],
                    ep_metrics['sampled_uncert_avg'], ep_metrics['rollout_uncert_avg'],
                ])
```

**변경 후:**
```python
                writer.writerow([
                    run_idx, episode_ct, total_step_ct, episode_reward,
                    ep_metrics['nominal_error_avg'], ep_metrics['residual_error_avg'],
                    ep_metrics['dhat_norm_avg'], ep_metrics['uncertainty_avg'],
                    ep_metrics['res_net_loss'], ep_metrics['rbf_loss'],
                    ep_metrics['td_loss_avg'], ep_metrics['episode_length'],
                    ep_metrics['expl_noise'], ep_metrics['buffer_uncert_avg'],
                    ep_metrics['sampled_uncert_avg'], ep_metrics['rollout_uncert_avg'],
                    ep_metrics['rollout_pass_rate'], ep_metrics['rollout_avg_horizon'],
                    ep_metrics['rbf_calib_ratio'], ep_metrics['rbf_calib_corr'],
                ])
```

---

## 구현 후 검증

학습 시작 후 `warm_start_samples`(10000 스텝) 이후부터 새 컬럼이 채워진다.
그 전까지는 `NaN`이 정상이다.

### 정상 범위 기준

| 지표 | 정상 범위 | 문제 상황 |
|---|---|---|
| `rollout_pass_rate` | 0.3 ~ 0.9 | ≈1.0 이면 threshold 너무 높음, ≈0.0 이면 너무 낮음 |
| `rollout_avg_horizon` | 2 ~ 8 (max=10) | ≈1.0 이면 h>0에서 즉시 차단, ≈10.0 이면 필터 미작동 |
| `rbf_calib_ratio` | 0.7 ~ 1.3 | <<1.0 이면 RBF under-predict (필터 무의미), >>1.0 이면 over-predict |
| `rbf_calib_corr` | > 0.5 | 낮으면 RBF가 고-uncertainty 상태를 식별 못함 |

### CSV 컬럼 확인 명령

```bash
head -1 results/<실험폴더>/seed_1_progress.csv
```

4개 컬럼(`rollout_pass_rate`, `rollout_avg_horizon`, `rbf_calib_ratio`, `rbf_calib_corr`)이 헤더에 있으면 정상.
