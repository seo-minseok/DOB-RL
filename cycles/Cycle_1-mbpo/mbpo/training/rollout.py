"""
rollout.py — MBPO model rollout 및 혼합 배치 샘플링

generate_model_rollout  : 학습된 Ensemble로 가상 transition 생성 → model_buffer 채움
sample_mixed_minibatch  : real + model buffer 혼합 샘플
compute_rollout_horizon : rollout_schedule에 따른 horizon 계산
"""
import numpy as np
import torch

from ..models.ensemble_model import EnsembleModel
from ..models.actor_network import ActorNetwork
from ..utils.buffer import ReplayBuffer
from ..envs.hopper_utils import reward_is_done_function


def compute_rollout_horizon(total_step_ct: int, epoch_length: int,
                            rollout_schedule: list) -> int:
    """
    rollout_schedule = [min_epoch, max_epoch, min_horizon, max_horizon]
    epoch-based 선형 보간.
    """
    min_ep, max_ep, min_h, max_h = rollout_schedule
    epoch = total_step_ct // epoch_length
    if epoch <= min_ep:
        return min_h
    if epoch >= max_ep:
        return max_h
    frac = (epoch - min_ep) / (max_ep - min_ep)
    return int(min_h + frac * (max_h - min_h))


def compute_model_pool_size(rollout_horizon: int, rollout_batch_size: int,
                            epoch_length: int, rollout_interval: int,
                            model_retain_epochs: int) -> int:
    """MBPO 원본 공식: retain_epochs * horizon * rollouts_per_epoch."""
    rollouts_per_epoch = rollout_batch_size * epoch_length / rollout_interval
    model_steps_per_epoch = int(rollout_horizon * rollouts_per_epoch)
    return model_retain_epochs * model_steps_per_epoch


def generate_model_rollout(real_buffer: ReplayBuffer, model_buffer: ReplayBuffer,
                            model: EnsembleModel, actor: ActorNetwork,
                            rollout_batch_size: int, rollout_horizon: int,
                            deterministic: bool = False,
                            device: torch.device = None) -> dict:
    """
    real_buffer에서 obs를 샘플링하여 model로 horizon 스텝 rollout 후 model_buffer에 저장.
    Returns rollout stats dict.
    """
    if device is None:
        device = next(actor.parameters()).device

    # rollout 시작 obs 샘플
    idx = np.random.randint(0, real_buffer.length, size=rollout_batch_size)
    obs = real_buffer.obs[idx].copy()   # (B, obs_dim)

    steps_added = []
    model.eval()
    actor.eval()

    for h in range(rollout_horizon):
        # Actor로 action 선택 (stochastic)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            act_t, _ = actor.sample(obs_t)
        act = act_t.cpu().numpy()   # (B, act_dim)

        # Ensemble prediction
        inputs = np.concatenate([obs, act], axis=-1).astype(np.float32)
        mean_np, var_np = model.predict(inputs, factored=True)  # (E, B, out_dim)
        std_np = np.sqrt(var_np)

        # 각 샘플에 대해 elite 중 하나를 무작위 선택
        B = obs.shape[0]
        model_inds = model.random_elite_inds(B)   # (B,)
        batch_inds = np.arange(B)

        if deterministic:
            samples = mean_np[model_inds, batch_inds]
        else:
            noise   = np.random.standard_normal(mean_np.shape)
            samples = mean_np + noise * std_np
            samples = samples[model_inds, batch_inds]   # (B, out_dim)

        # Parse predictions: first dim = reward, rest = delta_obs
        pred_rew   = samples[:, 0]                 # (B,)
        delta_obs  = samples[:, 1:]                # (B, obs_dim)
        next_obs   = obs + delta_obs               # (B, obs_dim)

        # Termination via synthetic reward
        _, is_done = reward_is_done_function(obs, act, next_obs)   # (B,)

        model_buffer.store_batch(
            obs_b      = obs,
            act_b      = act,
            next_obs_b = next_obs,
            rew_b      = pred_rew,
            done_b     = is_done,
        )
        steps_added.append(B)

        # 생존한 샘플만 다음 스텝으로
        nonterm_mask = ~is_done
        if nonterm_mask.sum() == 0:
            print(f'[ Rollout ] Breaking early at h={h}: all terminated')
            break
        obs = next_obs[nonterm_mask]

    mean_length = sum(steps_added) / rollout_batch_size
    stats = {
        'rollout_steps_added': sum(steps_added),
        'rollout_mean_length': mean_length,
        'model_pool_size':     model_buffer.length,
    }
    print(f'[ Rollout ] horizon={rollout_horizon} | added={sum(steps_added):.1e} '
          f'| pool={model_buffer.length:.1e} | mean_len={mean_length:.2f}')
    return stats


def sample_mixed_minibatch(model_trained: bool, real_ratio: float,
                            mini_batch_size: int,
                            real_buffer: ReplayBuffer,
                            model_buffer: ReplayBuffer):
    """real + model buffer 혼합 샘플."""
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
