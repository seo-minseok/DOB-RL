"""
trainer.py — train_MBPO_core 메인 루프 (Gymnasium Hopper-v5, SAC + Ensemble)

MBPO 알고리즘:
  - warm_start_steps: 무작위 탐색
  - 매 rollout_interval 스텝: Ensemble 학습 + model rollout
  - 매 스텝: n_train_repeat 번 SAC 업데이트
  - real_ratio=0.05 혼합 배치
"""
import csv
import dataclasses
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from .config import MBPOConfig
from .model_learning import train_ensemble
from .rollout import (
    generate_model_rollout, sample_mixed_minibatch,
    compute_rollout_horizon, compute_model_pool_size,
)
from ..models import ActorNetwork, QNetwork, EnsembleModel
from ..utils.buffer import ReplayBuffer
from ..envs.hopper_utils import make_hopper_env, reset_env, step_env

NUM_OBS = 11
NUM_ACT = 3


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    for p, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def _evaluate_policy(actor: nn.Module, eval_episodes: int,
                     max_steps: int, device: torch.device) -> float:
    eval_env = make_hopper_env()
    total = 0.0
    actor.eval()
    for _ in range(eval_episodes):
        obs    = np.array(reset_env(eval_env), dtype=np.float32)
        ep_rew = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                action = actor(
                    torch.tensor(obs, device=device).unsqueeze(0)
                ).cpu().numpy().flatten()
            action = np.clip(action, -1.0, 1.0)
            next_obs_raw, env_reward, terminated, truncated, _ = step_env(eval_env, action)
            ep_rew += float(env_reward)
            obs = np.array(next_obs_raw, dtype=np.float32)
            if terminated or truncated:
                break
        total += ep_rew
    eval_env.close()
    actor.train()
    return total / eval_episodes


_CSV_KEYS = [
    'td_loss_avg', 'q_pred_avg', 'target_q_avg', 'alpha',
    'model_val_loss', 'model_error_avg', 'rollout_mean_length', 'model_pool_size',
    'episode_length', 'env_return', 'eval_reward_avg',
]


def train_MBPO_core(run_idx: int,
                    result_queue=None,
                    checkpoint_dir: str = '.',
                    resume: bool = False,
                    cfg: MBPOConfig = None,
                    results_dir: str = None,
                    device: torch.device = None):
    """
    Args
    ----
    run_idx        : random seed (1-based)
    result_queue   : multiprocessing.Queue
    checkpoint_dir : 체크포인트 저장 디렉토리
    resume         : True이면 기존 체크포인트에서 재개
    cfg            : MBPOConfig 인스턴스
    device         : torch.device

    Returns
    -------
    episode_cumulative_reward_vector : list[float]
    episode_step_vector              : list[int]
    episode_metrics_list             : list[dict]
    """
    if cfg is None:
        cfg = MBPOConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[Seed {run_idx}] device={device}')

    np.random.seed(run_idx)
    torch.manual_seed(run_idx)

    env = make_hopper_env()

    # --- SAC Networks ---
    actor          = ActorNetwork(NUM_OBS, NUM_ACT).to(device)
    critic1        = QNetwork(NUM_OBS, NUM_ACT).to(device)
    target_critic1 = deepcopy(critic1)
    critic2        = QNetwork(NUM_OBS, NUM_ACT).to(device)
    target_critic2 = deepcopy(critic2)

    actor_opt   = optim.Adam(actor.parameters(),   lr=cfg.lr)
    critic1_opt = optim.Adam(critic1.parameters(), lr=cfg.lr)
    critic2_opt = optim.Adam(critic2.parameters(), lr=cfg.lr)

    log_alpha = torch.tensor(np.log(cfg.init_alpha), dtype=torch.float32,
                             device=device, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=cfg.lr_alpha)
    target_entropy = cfg.target_entropy

    # --- Ensemble Dynamics Model ---
    ensemble_model = EnsembleModel(
        obs_dim=NUM_OBS, act_dim=NUM_ACT, rew_dim=1,
        hidden_dim=cfg.hidden_dim,
        num_networks=cfg.num_networks,
        num_elites=cfg.num_elites,
    ).to(device)

    # --- Buffers (CPU) ---
    real_buffer  = ReplayBuffer(cfg.buffer_size, NUM_OBS, NUM_ACT)

    # model_buffer: 최대 크기 = model_retain_epochs * rollout_batch_size * max_horizon * (epoch_length / rollout_interval)
    _max_horizon  = cfg.rollout_schedule[3]
    _max_pool     = compute_model_pool_size(
        _max_horizon, cfg.rollout_batch_size,
        cfg.epoch_length, cfg.rollout_interval, cfg.model_retain_epochs
    )
    model_buffer = ReplayBuffer(max(int(_max_pool), cfg.rollout_batch_size), NUM_OBS, NUM_ACT)

    episode_cumulative_reward_vector = []
    episode_step_vector              = []
    episode_metrics_list             = []
    total_step_ct                    = 0
    episode_ct                       = 0
    model_trained_at_least_once      = False
    best_avg_score                   = -float('inf')
    last_model_val_loss              = float('nan')
    last_rollout_stats               = {}

    # --- Config-specific checkpoint subdir ---
    if results_dir is not None:
        ckpt_subdir = os.path.join(checkpoint_dir, os.path.basename(results_dir))
    else:
        ckpt_subdir = checkpoint_dir
    os.makedirs(ckpt_subdir, exist_ok=True)
    _config_path = os.path.join(ckpt_subdir, 'config.json')
    if not os.path.exists(_config_path):
        with open(_config_path, 'w') as _f:
            json.dump(dataclasses.asdict(cfg), _f, indent=2)

    # --- Resume ---
    checkpoint_path = os.path.join(ckpt_subdir, f'Champion_Seed{run_idx}_BestModel.pt')
    resume_path     = os.path.join(ckpt_subdir, f'Resume_Seed{run_idx}_Latest.pt')
    if resume:
        _ckpt = resume_path if os.path.exists(resume_path) else (
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )
        if _ckpt is not None:
            ck = torch.load(_ckpt, weights_only=False, map_location=device)
            actor.load_state_dict(ck['actor'])
            critic1.load_state_dict(ck['critic1'])
            target_critic1.load_state_dict(ck['target_critic1'])
            critic2.load_state_dict(ck['critic2'])
            target_critic2.load_state_dict(ck['target_critic2'])
            ensemble_model.load_state_dict(ck['ensemble_model'])
            actor_opt.load_state_dict(ck['actor_opt'])
            critic1_opt.load_state_dict(ck['critic1_opt'])
            critic2_opt.load_state_dict(ck['critic2_opt'])
            if 'log_alpha' in ck:
                log_alpha.data.copy_(ck['log_alpha'])
            if 'alpha_opt' in ck:
                alpha_opt.load_state_dict(ck['alpha_opt'])
            episode_ct                  = ck['episode']
            total_step_ct               = ck['total_steps']
            best_avg_score              = ck['best_avg_score']
            model_trained_at_least_once = ck['model_trained_at_least_once']
            episode_cumulative_reward_vector = ck['rewards']
            episode_step_vector              = ck['steps']
            print(f'[Seed {run_idx}] Resumed from {os.path.basename(_ckpt)} '
                  f'episode {ck["episode"]} (total_steps={total_step_ct})')

    # --- Main Training Loop ---
    _csv_new_session = not resume
    while total_step_ct < cfg.total_steps:
        episode_ct += 1

        obs            = np.array(reset_env(env), dtype=np.float32)
        obs_t          = torch.tensor(obs, device=device).unsqueeze(0)
        episode_reward = 0.0
        ep_td_losses   = []
        ep_q_preds     = []
        ep_tgt_qs      = []
        ep_env_rewards = []
        ep_model_errors = []
        step_ct        = 0

        for step_ct in range(1, cfg.max_steps_per_ep + 1):
            total_step_ct += 1

            # Action selection
            if total_step_ct <= cfg.warm_start_steps:
                action = np.random.uniform(-1.0, 1.0, size=(NUM_ACT,)).astype(np.float32)
            else:
                actor.eval()
                with torch.no_grad():
                    action, _ = actor.sample(obs_t)
                    action = action.cpu().numpy().flatten()
                actor.train()

            next_obs_raw, env_reward, terminated, truncated, _ = step_env(env, action)
            is_done  = terminated or truncated
            next_obs = np.array(next_obs_raw, dtype=np.float32)

            real_buffer.store(
                obs=obs, act=action, next_obs=next_obs,
                rew=np.float32(env_reward), done=terminated,
            )

            ep_env_rewards.append(float(env_reward))
            episode_reward += float(env_reward)

            if model_trained_at_least_once:
                inp = np.concatenate([obs, action], dtype=np.float32)[None]
                pred_mean, _ = ensemble_model.predict(inp, device=device)
                delta_obs_pred = pred_mean[0, 1:]
                delta_obs_real = next_obs - obs
                ep_model_errors.append(float(np.linalg.norm(delta_obs_real - delta_obs_pred)))

            obs   = next_obs
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)

            # [Model Training + Rollout] — every rollout_interval steps
            if (total_step_ct % cfg.rollout_interval == 0
                    and total_step_ct > cfg.warm_start_steps
                    and cfg.real_ratio < 1.0):

                print(f'[ MBPO ] step={total_step_ct} | epoch={total_step_ct // cfg.epoch_length} '
                      f'| buffer={real_buffer.length}')

                model_metrics = train_ensemble(
                    ensemble_model, real_buffer,
                    batch_size=cfg.model_batch_size,
                    holdout_ratio=cfg.holdout_ratio,
                    max_epochs_since_update=cfg.max_epochs_since_update,
                    device=device,
                )
                last_model_val_loss = model_metrics.get('val_loss', float('nan'))

                rollout_horizon = compute_rollout_horizon(
                    total_step_ct, cfg.epoch_length, cfg.rollout_schedule
                )
                last_rollout_stats = generate_model_rollout(
                    real_buffer, model_buffer, ensemble_model, actor,
                    rollout_batch_size=cfg.rollout_batch_size,
                    rollout_horizon=rollout_horizon,
                    device=device,
                )
                model_trained_at_least_once = True

            # [SAC Update] — n_train_repeat gradient steps per env step
            if total_step_ct > cfg.warm_start_steps:
                for _ in range(cfg.n_train_repeat):
                    if real_buffer.length < cfg.mini_batch_size:
                        break

                    s_obs, s_act, s_nxt, s_rew, s_done = sample_mixed_minibatch(
                        model_trained_at_least_once, cfg.real_ratio,
                        cfg.mini_batch_size, real_buffer, model_buffer
                    )

                    obs_bt  = torch.tensor(s_obs,  device=device)
                    nxt_bt  = torch.tensor(s_nxt,  device=device)
                    act_bt  = torch.tensor(s_act,  device=device)
                    rew_bt  = torch.tensor(s_rew,  device=device)
                    done_bt = torch.tensor(s_done, dtype=torch.bool, device=device)

                    alpha = log_alpha.exp().detach()

                    # Critic update
                    with torch.no_grad():
                        next_act, next_log_prob = actor.sample(nxt_bt)
                        q1_tgt = target_critic1(nxt_bt, next_act)
                        q2_tgt = target_critic2(nxt_bt, next_act)
                        target_q = (cfg.reward_scale * rew_bt
                                    + cfg.discount_factor * (
                                        torch.min(q1_tgt, q2_tgt)
                                        - alpha * next_log_prob.squeeze(-1)
                                    ))
                        target_q[done_bt] = cfg.reward_scale * rew_bt[done_bt]

                    q1_pred = critic1(obs_bt, act_bt)
                    q2_pred = critic2(obs_bt, act_bt)
                    loss_c1 = 0.5 * nn.functional.mse_loss(q1_pred, target_q)
                    loss_c2 = 0.5 * nn.functional.mse_loss(q2_pred, target_q)

                    critic1_opt.zero_grad()
                    loss_c1.backward()
                    critic1_opt.step()

                    critic2_opt.zero_grad()
                    loss_c2.backward()
                    critic2_opt.step()

                    ep_td_losses.append((loss_c1.item() + loss_c2.item()) * 0.5)
                    ep_q_preds.append(q1_pred.detach().mean().item())
                    ep_tgt_qs.append(target_q.detach().mean().item())

                    # Actor update
                    new_act, new_log_prob = actor.sample(obs_bt)
                    q1_new = critic1(obs_bt, new_act)
                    q2_new = critic2(obs_bt, new_act)
                    actor_loss = (alpha * new_log_prob.squeeze(-1)
                                  - torch.min(q1_new, q2_new)).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    # Alpha update
                    if cfg.auto_tune_alpha:
                        alpha_loss = -(log_alpha * (new_log_prob.detach() + target_entropy)).mean()
                        alpha_opt.zero_grad()
                        alpha_loss.backward()
                        alpha_opt.step()

                    _soft_update(critic1, target_critic1, cfg.tau)
                    _soft_update(critic2, target_critic2, cfg.tau)

            if is_done or total_step_ct >= cfg.total_steps:
                break

        # --- Evaluation ---
        ep_eval_reward_avg = float('nan')
        if episode_ct % cfg.eval_interval == 0:
            ep_eval_reward_avg = _evaluate_policy(
                actor, cfg.eval_episodes, cfg.max_steps_per_ep, device
            )

        ep_metrics = {
            'td_loss_avg':         float(np.mean(ep_td_losses))    if ep_td_losses    else float('nan'),
            'q_pred_avg':          float(np.mean(ep_q_preds))      if ep_q_preds      else float('nan'),
            'target_q_avg':        float(np.mean(ep_tgt_qs))       if ep_tgt_qs       else float('nan'),
            'alpha':               float(log_alpha.exp().item()),
            'model_val_loss':      last_model_val_loss,
            'model_error_avg':     float(np.mean(ep_model_errors)) if ep_model_errors else float('nan'),
            'rollout_mean_length': last_rollout_stats.get('rollout_mean_length', float('nan')),
            'model_pool_size':     float(model_buffer.length),
            'episode_length':      step_ct,
            'env_return':          float(np.sum(ep_env_rewards))   if ep_env_rewards  else float('nan'),
            'eval_reward_avg':     ep_eval_reward_avg,
        }
        episode_metrics_list.append(ep_metrics)
        episode_cumulative_reward_vector.append(episode_reward)
        episode_step_vector.append(total_step_ct)

        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            _csv_path = os.path.join(results_dir, f'seed_{run_idx}_progress.csv')
            open_mode = 'w' if _csv_new_session else 'a'
            write_hdr = open_mode == 'w' or not os.path.exists(_csv_path)
            _csv_new_session = False
            with open(_csv_path, open_mode, newline='') as _f:
                _w = csv.writer(_f)
                if write_hdr:
                    _w.writerow(['episode', 'total_steps', 'ep_steps', 'reward']
                                + _CSV_KEYS)
                _w.writerow([episode_ct, total_step_ct, step_ct, episode_reward]
                            + [ep_metrics[k] for k in _CSV_KEYS])

        if result_queue is not None:
            result_queue.put({
                'run_idx': run_idx, 'ep_idx': episode_ct,
                'reward': episode_reward, 'step': total_step_ct,
                'ep_steps': step_ct, **ep_metrics,
            })

        # --- Resume checkpoint ---
        torch.save({
            'actor':                     actor.state_dict(),
            'critic1':                   critic1.state_dict(),
            'target_critic1':            target_critic1.state_dict(),
            'critic2':                   critic2.state_dict(),
            'target_critic2':            target_critic2.state_dict(),
            'ensemble_model':            ensemble_model.state_dict(),
            'actor_opt':                 actor_opt.state_dict(),
            'critic1_opt':               critic1_opt.state_dict(),
            'critic2_opt':               critic2_opt.state_dict(),
            'log_alpha':                 log_alpha.data,
            'alpha_opt':                 alpha_opt.state_dict(),
            'episode':                   episode_ct,
            'total_steps':               total_step_ct,
            'best_avg_score':            best_avg_score,
            'model_trained_at_least_once': model_trained_at_least_once,
            'rewards':                   episode_cumulative_reward_vector,
            'steps':                     episode_step_vector,
        }, resume_path)

        if episode_ct % cfg.eval_interval == 0:
            print(f'[Seed {run_idx}] step={total_step_ct} ep={episode_ct} '
                  f'reward={episode_reward:.1f} eval={ep_eval_reward_avg:.1f} '
                  f'alpha={log_alpha.exp().item():.3f}')

        TARGET_SCORE = 2000.0
        if len(episode_cumulative_reward_vector) >= 10:
            current_avg = np.mean(episode_cumulative_reward_vector[-10:])
            if current_avg > best_avg_score and current_avg >= TARGET_SCORE:
                best_avg_score = current_avg
                torch.save({
                    'actor':          actor.state_dict(),
                    'critic1':        critic1.state_dict(),
                    'critic2':        critic2.state_dict(),
                    'ensemble_model': ensemble_model.state_dict(),
                    'log_alpha':      log_alpha.data,
                    'total_steps':    total_step_ct,
                    'episode':        episode_ct,
                }, checkpoint_path)
                print(f'[Seed {run_idx}] New best! Avg {current_avg:.1f} '
                      f'at ep {episode_ct} / step {total_step_ct}')

    env.close()
    return episode_cumulative_reward_vector, episode_step_vector, episode_metrics_list
