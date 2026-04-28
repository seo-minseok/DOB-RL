"""
rollout.py — Model-based rollout 및 혼합 미니배치 샘플링 (BipedalWalker, TD3)
연속 행동: argmax → actor 출력 + Gaussian 탐색 노이즈
"""
import numpy as np
import torch

from ..dynamics.dob import predict_next_obs_dob
from ..envs.bipedalwalker_utils import reward_is_done_function


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
    """
    max_horizon      = options['max_horizon_length']
    uncert_threshold = options['uncertainty_threshold']
    num_iter         = options['num_generate_sample_iteration']
    B                = options['mini_batch_size']
    noise_std        = max(rollout_noise, options['epsilon_min_model'])

    for _ in range(num_iter):
        if real_buffer.length < B:
            return model_buffer

        idx         = np.random.randint(0, real_buffer.length, size=B)
        current_obs = torch.tensor(real_buffer.obs[idx])   # (B, 24)
        alive_mask  = np.ones(B, dtype=bool)

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
            is_reliable      = uncert_mag_arr < uncert_threshold

            if h == 0:
                is_reliable[:] = True

            n_reliable = is_reliable.sum()
            if n_reliable == 0:
                break

            rel_obs  = valid_obs_np[is_reliable]
            rel_act  = valid_act[is_reliable]
            rel_next = predict_next_obs_dob(rel_obs, rel_act, res_net, p_nom, use_nominal)
            rel_rew, rel_done = reward_is_done_function(rel_next)

            model_buffer.store_batch(rel_obs, rel_act, rel_next, rel_rew, rel_done)

            alive_idx = np.where(alive_mask)[0]
            alive_mask[alive_idx[~is_reliable]] = False

            reliable_idx = alive_idx[is_reliable]
            alive_mask[reliable_idx[rel_done]] = False

            not_done = ~rel_done
            if not_done.any():
                to_update = reliable_idx[not_done]
                current_obs[to_update] = torch.tensor(rel_next[not_done])

    return model_buffer


def sample_mixed_minibatch(model_trained: bool, real_ratio: float,
                            mini_batch_size: int, real_buffer, model_buffer):
    """
    real_ratio=0.2 → 20% real, 80% model.
    """
    if model_trained and real_ratio < 1.0:
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
