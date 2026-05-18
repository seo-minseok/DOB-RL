"""
rollout.py — Model-based rollout 및 혼합 미니배치 샘플링 (BipedalWalker, TD3)
연속 행동: argmax → actor 출력 + Gaussian 탐색 노이즈
Cycle 5: h==0 무조건 통과 제거 — 모든 step에서 uncertainty gating 적용.
"""
import numpy as np
import torch

from ..dynamics.dob import predict_next_obs_dob
from ..envs.bipedalwalker_utils import reward_is_done_function


def generate_samples_dob(real_buffer, model_buffer,
                          res_net, uncert_model, actor,
                          rollout_noise: float, options: dict,
                          p_nom: dict, use_nominal: bool,
                          contact_net=None):
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
    pass_rate_vals = []   # 모든 step(h>=0)에서 수집
    horizon_counts = []   # iteration마다 trajectory별 완료 step 수

    for _ in range(num_iter):
        if real_buffer.length < B:
            break

        idx         = np.random.randint(0, real_buffer.length, size=B)
        current_obs = torch.tensor(real_buffer.obs[idx])   # (B, 14)
        alive_mask  = np.ones(B, dtype=bool)
        traj_horizon = np.zeros(B, dtype=np.float32)       # 각 trajectory의 완료 스텝

        for h in range(max_horizon):
            n_alive = alive_mask.sum()
            if n_alive == 0:
                break

            valid_obs = current_obs[alive_mask]   # (n_alive, 14)

            with torch.no_grad():
                valid_act_t = actor(valid_obs)    # (n_alive, 4)
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
            pass_rate_vals.append(float(is_reliable.mean()))

            n_reliable = is_reliable.sum()
            if n_reliable == 0:
                break

            rel_obs  = valid_obs_np[is_reliable]
            rel_act  = valid_act[is_reliable]
            rel_next = predict_next_obs_dob(rel_obs, rel_act, res_net, p_nom, use_nominal, contact_net)
            rel_rew, rel_done = reward_is_done_function(rel_obs, rel_act, rel_next)

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

    rollout_uncert_avg  = float(np.concatenate(uncert_mag_all).mean()) if uncert_mag_all else float('nan')
    rollout_pass_rate   = float(np.mean(pass_rate_vals))               if pass_rate_vals else float('nan')
    rollout_avg_horizon = float(np.mean(horizon_counts))               if horizon_counts else float('nan')
    return model_buffer, rollout_uncert_avg, rollout_pass_rate, rollout_avg_horizon


def sample_mixed_minibatch(model_trained: bool, real_ratio: float,
                            mini_batch_size: int, real_buffer, model_buffer):
    """
    real_ratio=0.2 → 20% real, 80% model.
    model_buffer가 비어있으면 real_buffer만 사용 (model_buffer.reset() 후 rollout pass가 0일 때 안전 처리).
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
