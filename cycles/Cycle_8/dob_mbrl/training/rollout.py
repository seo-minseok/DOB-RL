"""
rollout.py — Model-based rollout 및 혼합 미니배치 샘플링 (Hopper-v5, TD3)
연속 행동: actor 출력 + Gaussian 탐색 노이즈
uncertainty gating 없음 — RBF는 로깅 목적으로만 계산.
horizon 선형 스케줄: 1 → max_horizon_length (ep 20~200)
"""
import numpy as np
import torch

from ..dynamics.dob import predict_next_obs_dob
from ..envs.hopper_utils import reward_is_done_function


def generate_samples_dob(real_buffer, model_buffer,
                          res_net, uncert_model, actor,
                          rollout_noise: float, options: dict,
                          p_nom: dict, use_nominal: bool,
                          episode_ct: int = 1):
    """
    Model rollout — uncertainty gating 없이 scheduled horizon까지 무조건 rollout.
    RBF uncertainty는 로깅 목적으로만 계산된다 (truncation에 사용하지 않음).

    episode_ct   : 현재 에피소드 번호 (horizon 선형 스케줄에 사용)

    Returns
    -------
    model_buffer       : 갱신된 model_buffer
    rollout_uncert_avg : 롤아웃 전 스텝의 RBF 예측 uncertainty 평균 (로깅용)
    rollout_pass_rate  : 항상 1.0 (truncation 없음)
    rollout_avg_horizon: trajectory당 평균 완료 스텝 수
    """
    max_horizon = options['max_horizon_length']
    num_iter    = options['num_generate_sample_iteration']
    B           = options['mini_batch_size']
    noise_std   = max(rollout_noise, options['epsilon_min_model'])

    # 2단계 horizon 스케줄:
    #   ep [20, 2000]: 1 → max_horizon//2  (선형)
    #   ep [2000, num_episodes]: max_horizon//2 → max_horizon  (선형)
    _start_ep = 20
    _mid_ep   = 2000
    _end_ep   = options.get('num_episodes', 5000)
    half_h    = max_horizon // 2

    if episode_ct <= _start_ep:
        horizon_length = 1
    elif episode_ct <= _mid_ep:
        slope          = (half_h - 1) / (_mid_ep - _start_ep)
        horizon_length = int(1 + slope * (episode_ct - _start_ep))
    elif episode_ct < _end_ep:
        slope          = (max_horizon - half_h) / (_end_ep - _mid_ep)
        horizon_length = int(half_h + slope * (episode_ct - _mid_ep))
    else:
        horizon_length = max_horizon

    uncert_mag_all = []
    horizon_counts = []

    for _ in range(num_iter):
        if real_buffer.length < B:
            break

        idx          = np.random.randint(0, real_buffer.length, size=B)
        current_obs  = torch.tensor(real_buffer.obs[idx])   # (B, 11)
        alive_mask   = np.ones(B, dtype=bool)
        traj_horizon = np.zeros(B, dtype=np.float32)

        for _ in range(horizon_length):
            n_alive = alive_mask.sum()
            if n_alive == 0:
                break

            valid_obs    = current_obs[alive_mask]   # (n_alive, 11)
            valid_obs_np = valid_obs.cpu().numpy()

            with torch.no_grad():
                valid_act_t = actor(valid_obs)        # (n_alive, 3)
            valid_act = valid_act_t.numpy()

            noise     = np.random.normal(0.0, noise_std, size=valid_act.shape).astype(np.float32)
            valid_act = np.clip(valid_act + noise, -1.0, 1.0)

            # RBF uncertainty 로깅만 수행 (truncation 없음)
            inp_rbf = torch.tensor(np.concatenate([valid_obs_np, valid_act], axis=-1))
            with torch.no_grad():
                pred_uncert = uncert_model(inp_rbf).cpu().numpy()  # (n_alive, DOB_DIM)
            uncert_mag_all.append(np.linalg.norm(pred_uncert, axis=1))

            next_obs = predict_next_obs_dob(valid_obs_np, valid_act, res_net, p_nom, use_nominal)
            rew, done = reward_is_done_function(valid_obs_np, valid_act, next_obs)

            model_buffer.store_batch(valid_obs_np, valid_act, next_obs, rew, done)

            alive_idx = np.where(alive_mask)[0]
            traj_horizon[alive_idx] += 1.0
            alive_mask[alive_idx[done]] = False

            not_done = ~done
            if not_done.any():
                current_obs[alive_idx[not_done]] = torch.tensor(next_obs[not_done])

        horizon_counts.extend(traj_horizon.tolist())

    rollout_uncert_avg  = float(np.concatenate(uncert_mag_all).mean()) if uncert_mag_all else float('nan')
    rollout_pass_rate   = 1.0   # truncation 없음 — 항상 full rollout
    rollout_avg_horizon = float(np.mean(horizon_counts))               if horizon_counts else float('nan')
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
