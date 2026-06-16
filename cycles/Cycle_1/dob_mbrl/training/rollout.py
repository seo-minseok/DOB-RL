"""
rollout.py — Model-based rollout 및 혼합 미니배치 샘플링 (Hopper-v5, SAC)
SAC actor.sample()로 stochastic rollout 수행 (외부 탐색 노이즈 불필요).
horizon 스케줄: epoch 기반 선형 보간 (원본 MBPO와 동일).
  rollout_schedule = [min_epoch, max_epoch, min_horizon, max_horizon]
"""
import numpy as np
import torch

from ..dynamics.dob import predict_next_obs_dob
from ..envs.hopper_utils import reward_is_done_function



def compute_model_pool_size(rollout_horizon: int, rollout_batch_size: int,
                            epoch_length: int, rollout_interval: int,
                            model_retain_epochs: int) -> int:
    """MBPO 원본 공식: retain_epochs * horizon * rollouts_per_epoch."""
    rollouts_per_epoch = rollout_batch_size * epoch_length / rollout_interval
    model_steps_per_epoch = int(rollout_horizon * rollouts_per_epoch)
    return model_retain_epochs * model_steps_per_epoch


def generate_samples_dob(real_buffer, model_buffer,
                          res_net, uncert_model, actor,
                          options: dict,
                          p_nom: dict, use_nominal: bool):
    """
    Model rollout — SAC actor.sample()로 stochastic rollout.
    RBF uncertainty > uncertainty_threshold인 step은 즉시 truncate하고
    해당 step만 model buffer에 저장하지 않는다 (이전 step들은 이미 저장됨).

    Returns
    -------
    model_buffer       : 갱신된 model_buffer
    rollout_uncert_avg : 전 스텝의 RBF 예측 uncertainty 평균 (로깅용)
    rollout_pass_rate  : uncertainty 기준 통과 비율 (passed / total evaluated)
    rollout_avg_horizon: trajectory당 평균 완료 스텝 수
    """
    num_iter              = options['num_generate_sample_iteration']
    B                     = options['rollout_batch_size']
    horizon_length        = options['rollout_horizon']
    uncertainty_threshold = options['uncertainty_threshold']

    uncert_mag_all = []
    horizon_counts = []
    n_total_steps  = 0
    n_passed_steps = 0

    for _ in range(num_iter):
        if real_buffer.length == 0:
            break

        _device     = next(actor.parameters()).device
        idx         = np.random.randint(0, real_buffer.length, size=B)
        current_obs = torch.tensor(real_buffer.obs[idx], device=_device)
        alive_mask   = np.ones(B, dtype=bool)
        traj_horizon = np.zeros(B, dtype=np.float32)
        first_step   = np.ones(B, dtype=bool)   # 첫 step은 truncation 면제

        for _ in range(horizon_length):
            n_alive = alive_mask.sum()
            if n_alive == 0:
                break

            alive_idx    = np.where(alive_mask)[0]          # (n_alive,) → B 인덱스
            valid_obs    = current_obs[alive_mask]           # (n_alive, 11)
            valid_obs_np = valid_obs.cpu().numpy()

            with torch.no_grad():
                valid_act_t, _ = actor.sample(valid_obs)    # SAC stochastic sample
            valid_act = valid_act_t.cpu().numpy()

            # ── uncertainty 계산 및 truncation ──────────────────────────
            if uncert_model is not None:
                inp_rbf = torch.tensor(
                    np.concatenate([valid_obs_np, valid_act], axis=-1), device=_device)
                with torch.no_grad():
                    pred_uncert = uncert_model(inp_rbf).cpu().numpy()   # (n_alive, DOB_DIM)
                uncert_mag = np.linalg.norm(pred_uncert, axis=1)        # (n_alive,)
                uncert_mag_all.append(uncert_mag)

                uncert_truncate = uncert_mag > uncertainty_threshold    # (n_alive,) bool
                uncert_truncate[first_step[alive_idx]] = False          # 첫 step 면제

                n_total_steps  += n_alive
                n_passed_steps += int((~uncert_truncate).sum())

                # truncated trajectory 종료 (step도, 저장도 하지 않음)
                alive_mask[alive_idx[uncert_truncate]] = False
                pass_mask = ~uncert_truncate
            else:
                pass_mask = np.ones(n_alive, dtype=bool)

            # ── uncertainty 통과한 trajectory만 step ────────────────────
            if not pass_mask.any():
                continue

            pass_alive_idx = alive_idx[pass_mask]
            pass_obs_np    = valid_obs_np[pass_mask]
            pass_act       = valid_act[pass_mask]

            next_obs = predict_next_obs_dob(pass_obs_np, pass_act, res_net, p_nom, use_nominal)
            rew, done = reward_is_done_function(pass_obs_np, pass_act, next_obs)

            model_buffer.store_batch(pass_obs_np, pass_act, next_obs, rew, done)

            traj_horizon[pass_alive_idx] += 1.0
            first_step[pass_alive_idx] = False          # 첫 step 통과 후 면제 해제
            alive_mask[pass_alive_idx[done]] = False

            not_done = ~done
            if not_done.any():
                current_obs[pass_alive_idx[not_done]] = torch.tensor(
                    next_obs[not_done], device=_device)

        horizon_counts.extend(traj_horizon.tolist())

    rollout_uncert_avg  = float(np.concatenate(uncert_mag_all).mean()) if uncert_mag_all else float('nan')
    rollout_pass_rate   = (n_passed_steps / n_total_steps) if n_total_steps > 0 else 1.0
    rollout_avg_horizon = float(np.mean(horizon_counts)) if horizon_counts else float('nan')
    return model_buffer, rollout_uncert_avg, rollout_pass_rate, rollout_avg_horizon


def sample_mixed_minibatch(model_trained: bool, real_ratio: float,
                            mini_batch_size: int, real_buffer, model_buffer):
    """
    real_ratio=0.2 → 20% real, 80% model.
    model_buffer가 비어있으면 real_buffer만 사용.
    """
    if model_trained and real_ratio < 1.0 and model_buffer.length > 0:
        n_real  = int(np.ceil(real_ratio * mini_batch_size))
        n_model = mini_batch_size - n_real
        obs_r, act_r, nxt_r, rew_r, don_r = real_buffer.sample(n_real)
        obs_m, act_m, nxt_m, rew_m, don_m = model_buffer.sample(n_model)
        obs = np.concatenate([obs_r, obs_m], axis=0)
        act = np.concatenate([act_r, act_m], axis=0)
        nxt = np.concatenate([nxt_r, nxt_m], axis=0)
        rew = np.concatenate([rew_r, rew_m], axis=0)
        don = np.concatenate([don_r, don_m], axis=0)
    else:
        obs, act, nxt, rew, don = real_buffer.sample(mini_batch_size)
    return obs, act, nxt, rew, don
