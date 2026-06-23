"""
trainer.py — train_DOB_core 메인 루프 (Gymnasium Hopper-v5, SAC)
Twin critics + stochastic actor + auto-tune alpha (entropy temperature)
correct_position=False → DOB_DIM=6 (velocity-only, Cycle_1 호환)
correct_position=True  → DOB_DIM=11 (full state, Cycle_1-dob-mbrl 호환)
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

from .config import DOBMBRLConfig
from .model_learning import train_residual_dx_model_dob, train_uncertainty_rbf
from .rollout import (generate_samples_dob, sample_mixed_minibatch,
                      compute_model_pool_size)
from ..models import ActorNetwork, QNetwork, ResidualDxNet, NormalizedRBFModel
from ..utils.buffer import ReplayBufferDOB
from ..dynamics import (
    default_hopper_params, step_nominal_hopper,
    get_dob_matrices,
)
from ..envs.hopper_utils import (
    make_hopper_env, reset_env, step_env,
)

NUM_OBS = 11   # Hopper-v5 observation dim
NUM_ACT = 3    # Hopper-v5 action dim


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    for p, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def _evaluate_policy(actor: nn.Module, eval_episodes: int, max_steps: int,
                     device: torch.device) -> float:
    eval_env = make_hopper_env()
    total = 0.0
    for _ in range(eval_episodes):
        obs = np.array(reset_env(eval_env), dtype=np.float32)
        ep_rew = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                action = actor(torch.tensor(obs, device=device).unsqueeze(0)).cpu().numpy().flatten()
            action = np.clip(action, -1.0, 1.0)
            next_obs_raw, env_reward, terminated, truncated, _ = step_env(eval_env, action)
            next_obs = np.array(next_obs_raw, dtype=np.float32)
            ep_rew += float(env_reward)
            obs = next_obs
            if terminated or truncated:
                break
        total += ep_rew
    eval_env.close()
    return total / eval_episodes


_CSV_KEYS = [
    'nominal_error_avg', 'residual_error_avg', 'dhat_norm_avg', 'uncertainty_avg',
    'res_net_loss', 'rbf_loss', 'td_loss_avg',
    'buffer_uncert_avg', 'sampled_uncert_avg', 'fresh_uncert_avg', 'rollout_uncert_avg',
    'rollout_pass_rate', 'rollout_avg_horizon',
]


def _buf_state(buf: 'ReplayBufferDOB') -> dict:
    n = buf.length
    return {
        'obs': buf.obs[:n].copy(), 'next_obs': buf.next_obs[:n].copy(),
        'act': buf.act[:n].copy(), 'rew': buf.rew[:n].copy(),
        'done': buf.done[:n].copy(), 'dhat': buf.dhat[:n].copy(),
        'dx_nom': buf.dx_nom[:n].copy(), 'uncertainty': buf.uncertainty[:n].copy(),
        'index': buf.index, 'length': buf.length,
    }


def _restore_buf(buf: 'ReplayBufferDOB', state: dict) -> None:
    n = state['length']
    buf.obs[:n] = state['obs']; buf.next_obs[:n] = state['next_obs']
    buf.act[:n] = state['act']; buf.rew[:n] = state['rew']
    buf.done[:n] = state['done']; buf.dhat[:n] = state['dhat']
    buf.dx_nom[:n] = state['dx_nom']; buf.uncertainty[:n] = state['uncertainty']
    buf.index = state['index']; buf.length = state['length']


def train_DOB_core(run_idx: int,
                   result_queue=None,
                   checkpoint_dir: str = '.',
                   resume: bool = False,
                   cfg: DOBMBRLConfig = None,
                   results_dir: str = None,
                   device: torch.device = None):
    """
    Args
    ----
    run_idx        : random seed (1-based)
    result_queue   : multiprocessing.Queue
    checkpoint_dir : 체크포인트 저장 디렉토리
    resume         : True이면 기존 체크포인트에서 재개
    cfg            : DOBMBRLConfig 인스턴스 (None이면 기본값 사용)
    device         : torch.device (None이면 CUDA 자동 감지)

    Returns
    -------
    episode_cumulative_reward_vector : list[float]
    episode_step_vector              : list[int]
    episode_metrics_list             : list[dict]
    """
    if cfg is None:
        cfg = DOBMBRLConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dob_dim      = 6 if not cfg.correct_position else 11
    FPINV, F_MAT = get_dob_matrices(dob_dim)

    print(f'[Seed {run_idx}] device={device}')

    np.random.seed(run_idx)
    torch.manual_seed(run_idx)

    env = make_hopper_env()

    p_nom       = default_hopper_params()
    use_nominal = True
    use_dob     = True

    # --- SAC Networks ---
    actor          = ActorNetwork(NUM_OBS, NUM_ACT).to(device)
    critic1        = QNetwork(NUM_OBS, NUM_ACT).to(device)
    target_critic1 = deepcopy(critic1)
    critic2        = QNetwork(NUM_OBS, NUM_ACT).to(device)
    target_critic2 = deepcopy(critic2)

    actor_opt   = optim.Adam(actor.parameters(),   lr=cfg.lr_actor)
    critic1_opt = optim.Adam(critic1.parameters(), lr=cfg.lr_critic)
    critic2_opt = optim.Adam(critic2.parameters(), lr=cfg.lr_critic)

    # SAC temperature parameter (log_alpha for numerical stability)
    log_alpha = torch.tensor(np.log(cfg.init_alpha), dtype=torch.float32,
                             device=device, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=cfg.lr_alpha)
    target_entropy = cfg.target_entropy

    res_net     = ResidualDxNet(NUM_OBS, NUM_ACT, hidden=256, out_dim=dob_dim).to(device)
    res_net_opt = optim.Adam(res_net.parameters(), lr=cfg.lr_residual)

    uncert_model = NormalizedRBFModel(cfg.num_rbf_centers, cfg.rbf_width, cfg.rbf_initial_value, dob_dim=dob_dim).to(device)
    rbf_opt      = optim.SGD(uncert_model.parameters(), lr=cfg.lr_rbf, momentum=0.9)

    # --- Buffers (CPU) ---
    real_buffer  = ReplayBufferDOB(cfg.buffer_size, NUM_OBS, NUM_ACT, dob_dim=dob_dim)
    _max_pool    = compute_model_pool_size(
        cfg.rollout_horizon, cfg.rollout_batch_size,
        cfg.epoch_length, cfg.rollout_interval, cfg.model_retain_epochs,
    )
    model_buffer = ReplayBufferDOB(max(int(_max_pool), cfg.rollout_batch_size),
                                   NUM_OBS, NUM_ACT, dob_dim=dob_dim)

    sample_gen_options = {
        'uncertainty_threshold'         : cfg.uncertainty_threshold,
        'num_generate_sample_iteration' : cfg.num_generate_sample_iteration,
        'rollout_batch_size'            : cfg.rollout_batch_size,
        'rollout_horizon'               : cfg.rollout_horizon,
        'correct_position'              : cfg.correct_position,
    }

    episode_cumulative_reward_vector = []
    episode_step_vector              = []
    episode_metrics_list             = []
    total_step_ct                    = 0
    episode_ct                       = 0
    model_trained_at_least_once      = False
    best_avg_score                   = -float('inf')

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
        _ckpt_to_load = resume_path if os.path.exists(resume_path) else (
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )
        if _ckpt_to_load is not None:
            ckpt = torch.load(_ckpt_to_load, weights_only=False, map_location=device)
            actor.load_state_dict(ckpt['actor'])
            critic1.load_state_dict(ckpt['critic1'])
            target_critic1.load_state_dict(ckpt['target_critic1'])
            critic2.load_state_dict(ckpt['critic2'])
            target_critic2.load_state_dict(ckpt['target_critic2'])
            res_net.load_state_dict(ckpt['res_net'])
            uncert_model.load_state_dict(ckpt['uncert_model'])
            actor_opt.load_state_dict(ckpt['actor_opt'])
            critic1_opt.load_state_dict(ckpt['critic1_opt'])
            critic2_opt.load_state_dict(ckpt['critic2_opt'])
            res_net_opt.load_state_dict(ckpt['res_net_opt'])
            if 'rbf_opt' in ckpt:
                rbf_opt.load_state_dict(ckpt['rbf_opt'])
            if 'log_alpha' in ckpt:
                log_alpha.data.copy_(ckpt['log_alpha'])
            if 'alpha_opt' in ckpt:
                alpha_opt.load_state_dict(ckpt['alpha_opt'])
            episode_ct                  = ckpt['episode']
            total_step_ct               = ckpt['total_steps']
            best_avg_score              = ckpt['best_avg_score']
            model_trained_at_least_once = ckpt['model_trained_at_least_once']
            episode_cumulative_reward_vector = ckpt['rewards']
            episode_step_vector              = ckpt['steps']
            if 'real_buffer' in ckpt:
                _restore_buf(real_buffer, ckpt['real_buffer'])
            print(f'[Seed {run_idx}] Resumed from {os.path.basename(_ckpt_to_load)} '
                  f'episode {ckpt["episode"]} (total_steps={total_step_ct})')

    # --- Main Training Loop ---
    _csv_new_session = not resume
    while total_step_ct < cfg.total_steps:
        episode_ct += 1

        ep_res_net_loss        = float('nan')
        ep_rbf_loss            = float('nan')
        ep_buffer_uncert_avg   = float('nan')
        ep_sampled_uncert_avg  = float('nan')
        ep_fresh_uncert_avg    = float('nan')
        ep_rollout_uncert_avg  = float('nan')
        ep_rollout_pass_rate   = float('nan')
        ep_rollout_avg_horizon = float('nan')

        # [Phase 2] Episode Reset
        obs            = np.array(reset_env(env), dtype=np.float32)
        obs_t          = torch.tensor(obs, device=device).unsqueeze(0)
        episode_reward = 0.0
        dhat           = np.zeros(dob_dim, dtype=np.float32)
        ep_nominal_errors   = []
        ep_residual_errors  = []
        ep_dhat_norms       = []
        ep_uncertainty_mags = []
        ep_td_losses        = []
        ep_q_pred_avgs      = []
        ep_target_q_avgs    = []
        ep_env_rewards      = []

        # [Phase 3] Environment Interaction
        for step_ct in range(1, cfg.max_steps_per_ep + 1):
            total_step_ct += 1

            # Action selection: uniform random during warm-start, SAC stochastic otherwise
            if total_step_ct <= cfg.warm_start_steps:
                action = np.random.uniform(-1.0, 1.0, size=(NUM_ACT,)).astype(np.float32)
            else:
                with torch.no_grad():
                    action, _ = actor.sample(obs_t)
                    action = action.cpu().numpy().flatten()

            next_obs_raw, env_reward, terminated, truncated, _ = step_env(env, action)
            is_done = terminated or truncated
            next_obs = np.array(next_obs_raw, dtype=np.float32)

            # DOB online update
            dx_real = next_obs - obs
            if use_nominal:
                x_nom_next = step_nominal_hopper(
                    obs.reshape(1, -1),
                    action.reshape(1, -1),
                    p_nom
                ).flatten()
                dx_nom = x_nom_next - obs
            else:
                dx_nom = np.zeros_like(obs)

            if cfg.real_ratio < 1.0 and total_step_ct > cfg.warm_start_steps:
                with torch.no_grad():
                    inp_res = torch.tensor(
                        np.concatenate([obs, action], dtype=np.float32),
                        device=device,
                    ).unsqueeze(0)
                    dx_res = res_net(inp_res).cpu().numpy().flatten()   # (11,)
            else:
                dx_res = np.zeros(dob_dim, dtype=np.float32)

            e = dx_real - dx_nom
            if use_dob:
                dhat = cfg.dob_w * dx_res + (1.0 - cfg.dob_w) * (FPINV @ e)
            else:
                dhat = np.zeros(dob_dim, dtype=np.float32)

            uncertainty = FPINV @ e - dx_res

            ep_nominal_errors.append(np.linalg.norm(e))
            ep_residual_errors.append(np.linalg.norm(e - F_MAT @ dx_res))
            ep_dhat_norms.append(np.linalg.norm(dhat))
            ep_uncertainty_mags.append(np.linalg.norm(uncertainty))

            ep_env_rewards.append(float(env_reward))

            real_buffer.store(
                obs         = obs,
                act         = action,
                next_obs    = next_obs,
                rew         = np.float32(env_reward),
                done        = terminated,
                dhat        = dhat,
                dx_nom      = dx_nom,
                uncertainty = uncertainty,
            )

            episode_reward += float(env_reward)
            obs   = next_obs
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)

            # [Step-based Model Training & Rollout] — every rollout_interval steps
            if (total_step_ct % cfg.rollout_interval == 0
                    and total_step_ct > cfg.warm_start_steps
                    and cfg.real_ratio < 1.0):
                valid_len = real_buffer.length
                ep_buffer_uncert_avg = float(
                    np.linalg.norm(real_buffer.uncertainty[:valid_len], axis=1).mean()
                )
                ep_res_net_loss, ep_sampled_uncert_avg = train_residual_dx_model_dob(
                    res_net, res_net_opt, real_buffer,
                    cfg.mini_batch_size, cfg.num_epochs,
                    use_uncertainty_sampling=cfg.use_uncertainty_sampling,
                )
                if cfg.use_rbf:
                    ep_rbf_loss, ep_fresh_uncert_avg = train_uncertainty_rbf(
                        uncert_model, rbf_opt, real_buffer, res_net,
                        cfg.mini_batch_size, 5, fpinv=FPINV
                    )
                model_trained_at_least_once = True
                (model_buffer,
                 ep_rollout_uncert_avg,
                 ep_rollout_pass_rate,
                 ep_rollout_avg_horizon) = generate_samples_dob(
                    real_buffer, model_buffer, res_net,
                    uncert_model if cfg.use_rbf else None,
                    actor, sample_gen_options, p_nom, use_nominal,
                )

            # [Phase 4] SAC Agent Update
            if (step_ct % cfg.update_interval == 0) and total_step_ct > cfg.warm_start_steps:
                for _ in range(cfg.num_gradient_steps):
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
                        next_action, next_log_prob = actor.sample(nxt_bt)   # (B,3), (B,1)
                        q1_tgt = target_critic1(nxt_bt, next_action)
                        q2_tgt = target_critic2(nxt_bt, next_action)
                        target_q = rew_bt + cfg.discount_factor * (
                            torch.min(q1_tgt, q2_tgt) - alpha * next_log_prob.squeeze(-1)
                        )
                        target_q[done_bt] = rew_bt[done_bt]

                    q1_pred = critic1(obs_bt, act_bt)
                    q2_pred = critic2(obs_bt, act_bt)
                    loss_c1 = 0.5 * nn.functional.mse_loss(q1_pred, target_q)
                    loss_c2 = 0.5 * nn.functional.mse_loss(q2_pred, target_q)

                    critic1_opt.zero_grad()
                    loss_c1.backward()
                    nn.utils.clip_grad_norm_(critic1.parameters(), max_norm=10.0)
                    critic1_opt.step()

                    critic2_opt.zero_grad()
                    loss_c2.backward()
                    nn.utils.clip_grad_norm_(critic2.parameters(), max_norm=10.0)
                    critic2_opt.step()

                    ep_td_losses.append((loss_c1.item() + loss_c2.item()) * 0.5)
                    ep_q_pred_avgs.append(q1_pred.detach().mean().item())
                    ep_target_q_avgs.append(target_q.detach().mean().item())

                    # Actor update
                    new_action, new_log_prob = actor.sample(obs_bt)
                    q1_new = critic1(obs_bt, new_action)
                    q2_new = critic2(obs_bt, new_action)
                    actor_loss = (alpha * new_log_prob.squeeze(-1) - torch.min(q1_new, q2_new)).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    # Alpha (temperature) update
                    if cfg.auto_tune_alpha:
                        alpha_loss = -(log_alpha * (new_log_prob.detach() + target_entropy)).mean()
                        alpha_opt.zero_grad()
                        alpha_loss.backward()
                        alpha_opt.step()

                    # Soft update critics only (SAC has no target actor)
                    _soft_update(critic1, target_critic1, cfg.tau)
                    _soft_update(critic2, target_critic2, cfg.tau)

            if is_done:
                break

        ep_eval_reward_avg = float('nan')
        if episode_ct % cfg.eval_interval == 0:
            ep_eval_reward_avg = _evaluate_policy(
                actor, cfg.eval_episodes, cfg.max_steps_per_ep, device
            )

        ep_metrics = {
            'nominal_error_avg':   float(np.mean(ep_nominal_errors))   if ep_nominal_errors   else float('nan'),
            'residual_error_avg':  float(np.mean(ep_residual_errors))  if ep_residual_errors  else float('nan'),
            'dhat_norm_avg':       float(np.mean(ep_dhat_norms))       if ep_dhat_norms       else float('nan'),
            'uncertainty_avg':     float(np.mean(ep_uncertainty_mags)) if ep_uncertainty_mags else float('nan'),
            'res_net_loss':        ep_res_net_loss,
            'rbf_loss':            ep_rbf_loss,
            'td_loss_avg':         float(np.mean(ep_td_losses)) if ep_td_losses else float('nan'),
            'episode_length':      step_ct,
            'buffer_uncert_avg':   ep_buffer_uncert_avg,
            'sampled_uncert_avg':  ep_sampled_uncert_avg,
            'fresh_uncert_avg':    ep_fresh_uncert_avg,
            'rollout_uncert_avg':  ep_rollout_uncert_avg,
            'rollout_pass_rate':   ep_rollout_pass_rate,
            'rollout_avg_horizon': ep_rollout_avg_horizon,
            'q_pred_avg':          float(np.mean(ep_q_pred_avgs))   if ep_q_pred_avgs   else float('nan'),
            'target_q_avg':        float(np.mean(ep_target_q_avgs)) if ep_target_q_avgs else float('nan'),
            'env_return':          float(np.sum(ep_env_rewards))    if ep_env_rewards   else float('nan'),
            'eval_reward_avg':     ep_eval_reward_avg,
            'alpha':               float(log_alpha.exp().item()),
        }
        episode_metrics_list.append(ep_metrics)
        episode_cumulative_reward_vector.append(episode_reward)
        episode_step_vector.append(total_step_ct)

        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            _csv_path    = os.path.join(results_dir, f'seed_{run_idx}_progress.csv')
            open_mode    = 'w' if _csv_new_session else 'a'
            write_header = open_mode == 'w' or not os.path.exists(_csv_path)
            _csv_new_session = False
            with open(_csv_path, open_mode, newline='') as _f:
                _w = csv.writer(_f)
                if write_header:
                    _w.writerow(['episode', 'total_steps', 'eval_reward_avg', 'ep_steps', 'reward', 'env_return'] + _CSV_KEYS)
                _w.writerow([episode_ct, total_step_ct, ep_eval_reward_avg, step_ct, episode_reward, ep_metrics['env_return']]
                            + [ep_metrics[k] for k in _CSV_KEYS])

        if result_queue is not None:
            result_queue.put({
                'run_idx' : run_idx,
                'ep_idx'  : episode_ct,
                'reward'  : episode_reward,
                'step'    : total_step_ct,
                'ep_steps': step_ct,
                **ep_metrics,
            })

        # --- Resume checkpoint (every episode) ---
        torch.save({
            'actor':          actor.state_dict(),
            'critic1':        critic1.state_dict(),
            'target_critic1': target_critic1.state_dict(),
            'critic2':        critic2.state_dict(),
            'target_critic2': target_critic2.state_dict(),
            'res_net':        res_net.state_dict(),
            'uncert_model':   uncert_model.state_dict(),
            'actor_opt':      actor_opt.state_dict(),
            'critic1_opt':    critic1_opt.state_dict(),
            'critic2_opt':    critic2_opt.state_dict(),
            'res_net_opt':    res_net_opt.state_dict(),
            'rbf_opt':        rbf_opt.state_dict(),
            'log_alpha':      log_alpha.data,
            'alpha_opt':      alpha_opt.state_dict(),
            'episode':        episode_ct,
            'total_steps':    total_step_ct,
            'best_avg_score':              best_avg_score,
            'model_trained_at_least_once': model_trained_at_least_once,
            'rewards': episode_cumulative_reward_vector,
            'steps':   episode_step_vector,
            'real_buffer': _buf_state(real_buffer),
        }, resume_path)

        TARGET_SCORE = 2000.0
        if len(episode_cumulative_reward_vector) >= 10:
            current_avg = np.mean(episode_cumulative_reward_vector[-10:])
            if current_avg > best_avg_score and current_avg >= TARGET_SCORE:
                best_avg_score = current_avg
                torch.save({
                    'actor':        actor.state_dict(),
                    'critic1':      critic1.state_dict(),
                    'critic2':      critic2.state_dict(),
                    'res_net':      res_net.state_dict(),
                    'uncert_model': uncert_model.state_dict(),
                    'log_alpha':    log_alpha.data,
                    'total_steps':  total_step_ct,
                    'episode':      episode_ct,
                }, checkpoint_path)
                print(f'[Seed {run_idx}] New best! Avg {current_avg:.1f} '
                      f'at ep {episode_ct} / step {total_step_ct}')

    env.close()
    return episode_cumulative_reward_vector, episode_step_vector, episode_metrics_list
