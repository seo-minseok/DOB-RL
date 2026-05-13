"""
trainer_mbrl.py — Pure MBRL trainer for BipedalWalker (TD3 + ensemble transition models)
train_MBRL_core.py 방식: DOB/ResidualNet/RBF 없음, 단순 ensemble 전이 모델.

학습 구조 (train_MBRL_core.py 동일):
  Phase 1: 전이 모델 학습 + model rollout (에피소드 시작 전)
  Phase 2: 에피소드 리셋
  Phase 3: 환경 상호작용 + 실제 전이 저장
  Phase 4: TD3 critic/actor 업데이트 (update_interval 마다)
  → 에피소드 종료: metrics를 CSV에 저장, best checkpoint 갱신
"""
import csv
import dataclasses
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from .config import DOBMBRLConfig
from ..models import ActorNetwork, QNetwork, TransitionNetwork
from ..utils.buffer import ReplayBuffer
from ..envs.bipedalwalker_utils import (
    make_bipedalwalker_env, reset_env, step_env, reward_is_done_function,
)
from ..dynamics.constants import OBS_DIM, ACT_DIM


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    with torch.no_grad():
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def _train_transition_models(models, optimizers,
                              real_buffer: ReplayBuffer,
                              mini_batch_size: int,
                              num_epochs: int) -> float:
    """
    Ensemble 전이 모델 학습 (train_MBRL_core.py 방식).
    각 모델을 SGD로 num_epochs 학습.
    Target: next_obs - obs (delta obs).
    Returns 전 모델 평균 MSE loss.
    """
    total_loss = 0.0
    count = 0
    for model, opt in zip(models, optimizers):
        model.train()
        for _ in range(num_epochs):
            perm     = np.random.permutation(real_buffer.length)
            num_iter = real_buffer.length // mini_batch_size
            for it in range(num_iter):
                idx   = perm[it * mini_batch_size:(it + 1) * mini_batch_size]
                obs_t = torch.tensor(real_buffer.obs[idx])
                act_t = torch.tensor(real_buffer.act[idx])
                nxt_t = torch.tensor(real_buffer.next_obs[idx])

                opt.zero_grad()
                dx_pred = model(obs_t, act_t)
                dx_true = nxt_t - obs_t
                loss    = nn.functional.mse_loss(dx_pred, dx_true)
                loss.backward()
                opt.step()

                total_loss += loss.item()
                count      += 1
    return total_loss / max(1, count)


def _generate_samples_mbrl(real_buffer: ReplayBuffer,
                            model_buffer: ReplayBuffer,
                            transition_models,
                            actor: ActorNetwork,
                            options: dict,
                            episode_ct: int):
    """
    train_MBRL_core.py generate_samples와 동일한 구조.
    - horizon 선형 스케줄: 1 → max_horizon (에피소드 [20, 200])
    - 각 horizon step마다 model order 재셔플 (MATLAB randperm 방식)
    - uncertainty gating 없음
    - actor + Gaussian 탐색 노이즈로 행동 선택

    Returns (model_buffer, horizon_length)
    """
    num_models  = len(transition_models)
    min_horizon = 1
    max_horizon = options['max_horizon_length']
    start_ep    = 20
    end_ep      = 200
    num_iter    = options['num_generate_sample_iteration']
    B           = options['mini_batch_size']
    noise_std   = options['epsilon_min_model']

    # 선형 horizon 스케줄
    if episode_ct <= start_ep:
        horizon_length = min_horizon
    elif episode_ct >= end_ep:
        horizon_length = max_horizon
    else:
        slope          = (max_horizon - min_horizon) / (end_ep - start_ep)
        horizon_length = int(min_horizon + slope * (episode_ct - start_ep))

    for _ in range(num_iter):
        predicted_next_obs = [None] * num_models   # iteration마다 초기화

        for h in range(horizon_length):
            model_order = np.random.permutation(num_models)   # 매 horizon step 재셔플

            for mi, model_id in enumerate(model_order):
                if real_buffer.length < B:
                    continue

                model = transition_models[model_id]
                model.eval()

                # 시작 관측 선택
                if h == 0:
                    idx      = np.random.choice(real_buffer.length, B, replace=False)
                    curr_obs = torch.tensor(real_buffer.obs[idx])
                else:
                    curr_obs = predicted_next_obs[mi]
                    if curr_obs is None:   # 이전 step에서 skip된 경우 방어적 처리
                        idx      = np.random.choice(real_buffer.length, B, replace=False)
                        curr_obs = torch.tensor(real_buffer.obs[idx])

                # Actor로 행동 선택 + Gaussian 탐색 노이즈
                with torch.no_grad():
                    curr_act = actor(curr_obs)
                noise    = torch.randn_like(curr_act) * noise_std
                curr_act = (curr_act + noise).clamp(-1.0, 1.0)

                # 전이 예측
                with torch.no_grad():
                    dx        = model(curr_obs, curr_act)
                pred_next = curr_obs + dx

                pred_next_np = pred_next.numpy()
                rew, done    = reward_is_done_function(curr_obs.numpy(), curr_act.numpy(), pred_next_np)

                model_buffer.store_batch(
                    obs_b      = curr_obs.numpy(),
                    act_b      = curr_act.numpy(),
                    next_obs_b = pred_next_np,
                    rew_b      = rew,
                    done_b     = done,
                )
                predicted_next_obs[mi] = pred_next

    return model_buffer, horizon_length


def _sample_mixed_minibatch(model_trained: bool,
                             real_ratio: float,
                             mini_batch_size: int,
                             real_buffer: ReplayBuffer,
                             model_buffer: ReplayBuffer):
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


def train_MBRL_core(run_idx: int,
                    num_episodes: int,
                    checkpoint_dir: str = '.',
                    resume: bool = False,
                    cfg: DOBMBRLConfig = None,
                    results_dir: str = None):
    """
    Pure MBRL trainer: TD3 + ensemble transition models. DOB 없음.

    Parameters
    ----------
    run_idx        : 랜덤 시드 (1-based)
    num_episodes   : 총 학습 에피소드 수
    checkpoint_dir : 체크포인트 저장 디렉토리
    resume         : True이면 기존 체크포인트에서 재개
    cfg            : DOBMBRLConfig (None이면 기본값)
    results_dir    : per-episode CSV 저장 경로 (None이면 저장 안 함)

    Returns
    -------
    episode_rewards : list[float]
    episode_steps   : list[int]  (누적 총 스텝)
    """
    if cfg is None:
        cfg = DOBMBRLConfig()

    np.random.seed(run_idx)
    torch.manual_seed(run_idx)

    env = make_bipedalwalker_env()

    # --- TD3 Networks ---
    actor          = ActorNetwork(OBS_DIM, ACT_DIM)
    target_actor   = deepcopy(actor)
    actor_opt      = optim.Adam(actor.parameters(), lr=cfg.lr_actor)

    critic1        = QNetwork(OBS_DIM, ACT_DIM)
    target_critic1 = deepcopy(critic1)
    critic1_opt    = optim.Adam(critic1.parameters(), lr=cfg.lr_critic)

    critic2        = QNetwork(OBS_DIM, ACT_DIM)
    target_critic2 = deepcopy(critic2)
    critic2_opt    = optim.Adam(critic2.parameters(), lr=cfg.lr_critic)

    for p in target_actor.parameters():   p.requires_grad_(False)
    for p in target_critic1.parameters(): p.requires_grad_(False)
    for p in target_critic2.parameters(): p.requires_grad_(False)

    # --- Ensemble Transition Models (train_MBRL_core.py: 3 models, SGD/momentum) ---
    num_tm = 3
    transition_models = [
        TransitionNetwork(OBS_DIM, ACT_DIM, hidden=256)
        for _ in range(num_tm)
    ]
    model_optimizers = [
        optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)
        for m in transition_models
    ]

    # --- Buffers ---
    real_buffer  = ReplayBuffer(cfg.buffer_size, OBS_DIM, ACT_DIM)
    model_buffer = ReplayBuffer(cfg.buffer_size, OBS_DIM, ACT_DIM)

    sample_gen_options = {
        'max_horizon_length'            : cfg.max_horizon_length,
        'num_generate_sample_iteration' : cfg.num_generate_sample_iteration,
        'mini_batch_size'               : cfg.mini_batch_size,
        'epsilon_min_model'             : cfg.epsilon_min_model,
    }

    episode_rewards  = []
    episode_steps    = []
    total_step_ct    = 0
    total_grad_steps = 0
    model_trained    = False
    best_avg_score   = -float('inf')
    start_episode    = 1

    checkpoint_path = os.path.join(checkpoint_dir, f'MBRL_Seed{run_idx}_BestModel.pt')

    # --- Resume ---
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        actor.load_state_dict(ckpt['actor'])
        target_actor.load_state_dict(ckpt['actor'])
        critic1.load_state_dict(ckpt['critic1'])
        critic2.load_state_dict(ckpt['critic2'])
        target_critic1.load_state_dict(ckpt['critic1'])
        target_critic2.load_state_dict(ckpt['critic2'])
        for i, m in enumerate(transition_models):
            key = f'transition_model_{i}'
            if key in ckpt:
                m.load_state_dict(ckpt[key])
        start_episode    = ckpt['episode'] + 1
        total_step_ct    = ckpt['total_steps']
        total_grad_steps = ckpt.get('total_grad_steps', 0)
        print(f'[MBRL Seed {run_idx}] Resumed from episode {ckpt["episode"]} '
              f'(total_steps={total_step_ct})')

    # --- Main Training Loop ---
    for episode_ct in range(start_episode, num_episodes + 1):

        ep_model_loss = float('nan')
        ep_horizon    = 0

        # [Phase 1] Transition model training + model rollouts
        if real_buffer.length > cfg.mini_batch_size and total_step_ct > cfg.warm_start_samples:
            if cfg.real_ratio < 1.0:
                ep_model_loss = _train_transition_models(
                    transition_models, model_optimizers,
                    real_buffer, cfg.mini_batch_size, cfg.num_epochs,
                )
                model_buffer, ep_horizon = _generate_samples_mbrl(
                    real_buffer, model_buffer, transition_models,
                    actor, sample_gen_options, episode_ct,
                )
                model_trained = True

        # [Phase 2] Episode reset
        obs            = reset_env(env)
        obs_t          = torch.tensor(obs).unsqueeze(0)
        episode_reward = 0.0
        ep_td_losses   = []
        ep_q1_vals     = []
        ep_q2_vals     = []
        ep_target_q    = []

        # [Phase 3] Environment interaction
        for step_ct in range(1, cfg.max_steps_per_ep + 1):
            total_step_ct += 1

            if total_step_ct <= cfg.warm_start_samples:
                action = np.random.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)
            else:
                with torch.no_grad():
                    action = actor(obs_t).cpu().numpy().flatten()
                noise  = np.random.normal(0.0, cfg.expl_noise, size=ACT_DIM).astype(np.float32)
                action = np.clip(action + noise, -1.0, 1.0)

            next_obs, env_reward, is_done, _ = step_env(env, action)
            reward = float(env_reward)

            real_buffer.store(
                obs      = obs,
                act      = action,
                next_obs = next_obs,
                rew      = np.float32(reward),
                done     = is_done,
            )

            episode_reward += reward
            obs   = next_obs
            obs_t = torch.tensor(obs).unsqueeze(0)

            # [Phase 4] TD3 update
            if (step_ct % cfg.update_interval == 0) and (total_step_ct > cfg.warm_start_samples):
                total_updates = cfg.num_gradient_steps * cfg.update_interval

                for _ in range(total_updates):
                    if real_buffer.length < cfg.mini_batch_size:
                        break
                    total_grad_steps += 1

                    s_obs, s_act, s_nxt, s_rew, s_done = _sample_mixed_minibatch(
                        model_trained, cfg.real_ratio,
                        cfg.mini_batch_size, real_buffer, model_buffer,
                    )

                    obs_bt  = torch.tensor(s_obs)
                    act_bt  = torch.tensor(s_act)
                    nxt_bt  = torch.tensor(s_nxt)
                    rew_bt  = torch.tensor(s_rew)
                    done_bt = torch.tensor(s_done, dtype=torch.bool)

                    # Critic targets (target policy noise + double Q)
                    with torch.no_grad():
                        noise_t = torch.clamp(
                            torch.randn_like(act_bt) * cfg.policy_noise,
                            -cfg.noise_clip, cfg.noise_clip,
                        )
                        tgt_act  = torch.clamp(target_actor(nxt_bt) + noise_t, -1.0, 1.0)
                        q1_next  = target_critic1(nxt_bt, tgt_act)
                        q2_next  = target_critic2(nxt_bt, tgt_act)
                        target_q = rew_bt + cfg.discount_factor * torch.min(q1_next, q2_next)
                        target_q[done_bt] = rew_bt[done_bt]

                    # Critic 1 update
                    q1_pred = critic1(obs_bt, act_bt)
                    loss_c1 = nn.functional.mse_loss(q1_pred, target_q)
                    critic1_opt.zero_grad()
                    loss_c1.backward()
                    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
                    critic1_opt.step()

                    # Critic 2 update
                    q2_pred = critic2(obs_bt, act_bt)
                    loss_c2 = nn.functional.mse_loss(q2_pred, target_q)
                    critic2_opt.zero_grad()
                    loss_c2.backward()
                    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
                    critic2_opt.step()

                    ep_td_losses.append((loss_c1.item() + loss_c2.item()) / 2.0)
                    ep_q1_vals.append(float(q1_pred.detach().mean()))
                    ep_q2_vals.append(float(q2_pred.detach().mean()))
                    ep_target_q.append(float(target_q.detach().mean()))

                    # Actor update (policy_delay 마다)
                    if total_grad_steps % cfg.policy_delay == 0:
                        actor_loss = -critic1(obs_bt, actor(obs_bt)).mean()
                        actor_opt.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                        actor_opt.step()

                        _soft_update(actor,   target_actor,   cfg.tau)
                        _soft_update(critic1, target_critic1, cfg.tau)
                        _soft_update(critic2, target_critic2, cfg.tau)

            if is_done:
                break

        # --- Episode metrics ---
        ep_metrics = {
            'td_loss_avg'       : float(np.mean(ep_td_losses)) if ep_td_losses else float('nan'),
            'model_loss_avg'    : ep_model_loss,
            'horizon_length'    : ep_horizon,
            'real_buffer_size'  : real_buffer.length,
            'model_buffer_size' : model_buffer.length,
            'q1_avg'            : float(np.mean(ep_q1_vals))  if ep_q1_vals  else float('nan'),
            'q2_avg'            : float(np.mean(ep_q2_vals))  if ep_q2_vals  else float('nan'),
            'target_q_avg'      : float(np.mean(ep_target_q)) if ep_target_q else float('nan'),
            'episode_length'    : step_ct,
        }

        episode_rewards.append(episode_reward)
        episode_steps.append(total_step_ct)

        # --- Per-episode incremental CSV save ---
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            csv_path     = os.path.join(results_dir, f'seed_{run_idx}_progress.csv')
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'w' if write_header else 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'seed', 'episode', 'total_steps', 'reward', 'episode_length',
                        'td_loss_avg', 'model_loss_avg', 'horizon_length',
                        'real_buffer_size', 'model_buffer_size',
                        'q1_avg', 'q2_avg', 'target_q_avg',
                    ])
                writer.writerow([
                    run_idx, episode_ct, total_step_ct, episode_reward,
                    ep_metrics['episode_length'],
                    ep_metrics['td_loss_avg'],
                    ep_metrics['model_loss_avg'],
                    ep_metrics['horizon_length'],
                    ep_metrics['real_buffer_size'],
                    ep_metrics['model_buffer_size'],
                    ep_metrics['q1_avg'],
                    ep_metrics['q2_avg'],
                    ep_metrics['target_q_avg'],
                ])

        # --- Checkpoint (10-ep avg 기준, 새 best마다 저장) ---
        if len(episode_rewards) >= 10:
            current_avg = np.mean(episode_rewards[-10:])
            if current_avg > best_avg_score:
                best_avg_score = current_avg
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_dict = {
                    'actor'           : actor.state_dict(),
                    'critic1'         : critic1.state_dict(),
                    'critic2'         : critic2.state_dict(),
                    'total_steps'     : total_step_ct,
                    'total_grad_steps': total_grad_steps,
                    'episode'         : episode_ct,
                    'config'          : dataclasses.asdict(cfg),
                }
                for i, m in enumerate(transition_models):
                    save_dict[f'transition_model_{i}'] = m.state_dict()
                torch.save(save_dict, checkpoint_path)
                print(f'[MBRL Seed {run_idx}] New best! Avg {current_avg:.1f} '
                      f'at ep {episode_ct} / step {total_step_ct}')

    env.close()
    return episode_rewards, episode_steps
