"""
trainer.py — train_DOB_core 메인 루프 (Gymnasium Hopper-v5, TD3)
Twin critics + delayed actor update + target policy smoothing
DOB_DIM=11 (전체 state 추적, FPINV = F_MAT = I_11)
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
from .rollout import generate_samples_dob, sample_mixed_minibatch
from ..models import ActorNetwork, QNetwork, ResidualDxNet, NormalizedRBFModel
from ..utils.buffer import ReplayBufferDOB
from ..dynamics import (
    default_hopper_params, step_nominal_hopper,
    FPINV, F_MAT, DOB_DIM,
)
from ..envs.hopper_utils import (
    make_hopper_env, reset_env, step_env, reward_is_done_function,
)

NUM_OBS = 11   # Hopper-v5 observation dim
NUM_ACT = 3    # Hopper-v5 action dim


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    for p, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def _evaluate_policy(actor: nn.Module, eval_episodes: int, max_steps: int) -> float:
    eval_env = make_hopper_env()
    total = 0.0
    for _ in range(eval_episodes):
        obs = np.array(reset_env(eval_env), dtype=np.float32)
        ep_rew = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                action = actor(torch.tensor(obs).unsqueeze(0)).cpu().numpy().flatten()
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
                   results_dir: str = None):
    """
    Args
    ----
    run_idx        : random seed (1-based)
    result_queue   : multiprocessing.Queue
    checkpoint_dir : 체크포인트 저장 디렉토리
    resume         : True이면 기존 체크포인트에서 재개
    cfg            : DOBMBRLConfig 인스턴스 (None이면 기본값 사용)

    Returns
    -------
    episode_cumulative_reward_vector : list[float]
    episode_step_vector              : list[int]
    episode_metrics_list             : list[dict]
    """
    if cfg is None:
        cfg = DOBMBRLConfig()

    np.random.seed(run_idx)
    torch.manual_seed(run_idx)

    env = make_hopper_env()

    p_nom       = default_hopper_params()
    use_nominal = True
    use_dob     = True

    # --- TD3 Networks ---
    actor         = ActorNetwork(NUM_OBS, NUM_ACT)
    target_actor  = deepcopy(actor)
    critic1       = QNetwork(NUM_OBS, NUM_ACT)
    target_critic1 = deepcopy(critic1)
    critic2       = QNetwork(NUM_OBS, NUM_ACT)
    target_critic2 = deepcopy(critic2)

    actor_opt   = optim.Adam(actor.parameters(),   lr=cfg.lr_actor)
    critic1_opt = optim.Adam(critic1.parameters(), lr=cfg.lr_critic)
    critic2_opt = optim.Adam(critic2.parameters(), lr=cfg.lr_critic)

    res_net     = ResidualDxNet(NUM_OBS, NUM_ACT, hidden=64, out_dim=DOB_DIM)
    res_net_opt = optim.SGD(res_net.parameters(), lr=cfg.lr_residual, momentum=0.9)

    uncert_model = NormalizedRBFModel(cfg.num_rbf_centers, cfg.rbf_width, cfg.rbf_initial_value)
    rbf_opt      = optim.SGD(uncert_model.parameters(), lr=cfg.lr_rbf, momentum=0.9)

    # --- Buffers ---
    real_buffer  = ReplayBufferDOB(cfg.buffer_size, NUM_OBS, NUM_ACT, dob_dim=DOB_DIM)
    model_buffer = ReplayBufferDOB(cfg.buffer_size, NUM_OBS, NUM_ACT, dob_dim=DOB_DIM)

    sample_gen_options = {
        'max_horizon_length'            : cfg.max_horizon_length,
        'uncertainty_threshold'         : cfg.uncertainty_threshold,
        'num_generate_sample_iteration' : cfg.num_generate_sample_iteration,
        'mini_batch_size'               : cfg.mini_batch_size,
        'num_observations'              : NUM_OBS,
        'epsilon_min_model'             : cfg.epsilon_min_model,
        'num_episodes'                  : cfg.rollout_end_episode,
    }

    episode_cumulative_reward_vector = []
    episode_step_vector              = []
    episode_metrics_list             = []
    total_step_ct                    = 0
    episode_ct                       = 0
    policy_update_ct                 = 0   # TD3 delayed actor update 카운터
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
            ckpt = torch.load(_ckpt_to_load, weights_only=False)
            actor.load_state_dict(ckpt['actor'])
            target_actor.load_state_dict(ckpt['target_actor'])
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
            rbf_opt.load_state_dict(ckpt['rbf_opt'])
            episode_ct                 = ckpt['episode']
            total_step_ct              = ckpt['total_steps']
            policy_update_ct           = ckpt['policy_update_ct']
            best_avg_score             = ckpt['best_avg_score']
            model_trained_at_least_once = ckpt['model_trained_at_least_once']
            episode_cumulative_reward_vector = ckpt['rewards']
            episode_step_vector              = ckpt['steps']
            if 'real_buffer' in ckpt:
                _restore_buf(real_buffer, ckpt['real_buffer'])
            print(f'[Seed {run_idx}] Resumed from {os.path.basename(_ckpt_to_load)} '
                  f'episode {ckpt["episode"]} (total_steps={total_step_ct})')

    # --- Main Training Loop ---
    _csv_new_session = not resume   # 새 학습: 첫 에피소드에서 덮어쓰기, resume: 항상 append
    while episode_ct < cfg.total_episodes:
        episode_ct += 1

        ep_res_net_loss        = float('nan')
        ep_rbf_loss            = float('nan')
        ep_buffer_uncert_avg   = float('nan')
        ep_sampled_uncert_avg  = float('nan')
        ep_fresh_uncert_avg    = float('nan')
        ep_rollout_uncert_avg  = float('nan')
        ep_rollout_pass_rate   = float('nan')
        ep_rollout_avg_horizon = float('nan')

        # [Phase 1] Model Training & Rollout
        if real_buffer.length > cfg.mini_batch_size and episode_ct > cfg.warm_start_episodes:
            if cfg.real_ratio < 1.0:
                valid_len = real_buffer.length
                ep_buffer_uncert_avg = float(
                    np.linalg.norm(real_buffer.uncertainty[:valid_len], axis=1).mean()
                )
                ep_res_net_loss, ep_sampled_uncert_avg = train_residual_dx_model_dob(
                    res_net, res_net_opt, real_buffer,
                    cfg.mini_batch_size, cfg.num_epochs,
                    use_uncertainty_sampling=cfg.use_uncertainty_sampling,
                )
                ep_rbf_loss, ep_fresh_uncert_avg = train_uncertainty_rbf(
                    uncert_model, rbf_opt, real_buffer, res_net,
                    cfg.mini_batch_size, 5
                )
                model_trained_at_least_once = True
                (model_buffer,
                 ep_rollout_uncert_avg,
                 ep_rollout_pass_rate,
                 ep_rollout_avg_horizon) = generate_samples_dob(
                    real_buffer, model_buffer, res_net, uncert_model,
                    actor, cfg.expl_noise, sample_gen_options, p_nom, use_nominal,
                    episode_ct=episode_ct,
                )

        # [Phase 2] Episode Reset
        obs            = np.array(reset_env(env), dtype=np.float32)
        obs_t          = torch.tensor(obs).unsqueeze(0)
        episode_reward = 0.0
        dhat           = np.zeros(DOB_DIM, dtype=np.float32)
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

            # Action selection: uniform random during warm-start, DDPG noise otherwise
            if episode_ct <= cfg.warm_start_episodes:
                action = np.random.uniform(-1.0, 1.0, size=(NUM_ACT,)).astype(np.float32)
            else:
                with torch.no_grad():
                    action = actor(obs_t).cpu().numpy().flatten()   # (3,)
                noise  = np.random.normal(0.0, cfg.expl_noise, size=action.shape).astype(np.float32)
                action = np.clip(action + noise, -1.0, 1.0)

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
                dx_nom = x_nom_next - obs   # zeros for null nominal
            else:
                dx_nom = np.zeros_like(obs)

            if cfg.real_ratio < 1.0 and episode_ct > cfg.warm_start_episodes:
                with torch.no_grad():
                    inp_res = torch.tensor(
                        np.concatenate([obs, action], dtype=np.float32)
                    ).unsqueeze(0)
                    dx_res = res_net(inp_res).cpu().numpy().flatten()   # (11,)
            else:
                dx_res = np.zeros(DOB_DIM, dtype=np.float32)

            e = dx_real - dx_nom                                    # (11,)
            if use_dob:
                dhat = cfg.dob_w * dx_res + (1.0 - cfg.dob_w) * (FPINV @ e)  # (11,)
            else:
                dhat = np.zeros(DOB_DIM, dtype=np.float32)

            uncertainty = FPINV @ e - dx_res                        # (11,)

            ep_nominal_errors.append(np.linalg.norm(e))
            ep_residual_errors.append(np.linalg.norm(e - F_MAT @ dx_res))
            ep_dhat_norms.append(np.linalg.norm(dhat))
            ep_uncertainty_mags.append(np.linalg.norm(uncertainty))

            shaped_rew, _ = reward_is_done_function(
                obs.reshape(1, -1), action.reshape(1, -1), next_obs.reshape(1, -1)
            )
            reward = float(shaped_rew[0])
            ep_env_rewards.append(float(env_reward))

            real_buffer.store(
                obs         = obs,
                act         = action,
                next_obs    = next_obs,
                rew         = np.float32(reward),
                done        = terminated,
                dhat        = dhat,
                dx_nom      = dx_nom,
                uncertainty = uncertainty,
            )

            episode_reward += reward
            obs   = next_obs
            obs_t = torch.tensor(obs).unsqueeze(0)

            # [Phase 4] TD3 Agent Update
            if (step_ct % cfg.update_interval == 0) and episode_ct > cfg.warm_start_episodes:
                for _ in range(cfg.num_gradient_steps):
                    if real_buffer.length < cfg.mini_batch_size:
                        break

                    s_obs, s_act, s_nxt, s_rew, s_done = sample_mixed_minibatch(
                        model_trained_at_least_once, cfg.real_ratio,
                        cfg.mini_batch_size, real_buffer, model_buffer
                    )

                    obs_bt  = torch.tensor(s_obs)
                    nxt_bt  = torch.tensor(s_nxt)
                    act_bt  = torch.tensor(s_act)
                    rew_bt  = torch.tensor(s_rew)
                    done_bt = torch.tensor(s_done, dtype=torch.bool)

                    with torch.no_grad():
                        noise    = (torch.randn_like(act_bt) * cfg.target_noise).clamp(
                                       -cfg.target_noise_clip, cfg.target_noise_clip)
                        tgt_act  = (target_actor(nxt_bt) + noise).clamp(-1.0, 1.0)
                        q1_tgt   = target_critic1(nxt_bt, tgt_act)
                        q2_tgt   = target_critic2(nxt_bt, tgt_act)
                        target_q = rew_bt + cfg.discount_factor * torch.min(q1_tgt, q2_tgt)
                        target_q[done_bt] = rew_bt[done_bt]

                    q1_pred = critic1(obs_bt, act_bt)
                    q2_pred = critic2(obs_bt, act_bt)
                    loss_c1 = nn.functional.mse_loss(q1_pred, target_q)
                    loss_c2 = nn.functional.mse_loss(q2_pred, target_q)

                    critic1_opt.zero_grad()
                    loss_c1.backward()
                    torch.nn.utils.clip_grad_value_(critic1.parameters(), 1.0)
                    critic1_opt.step()

                    critic2_opt.zero_grad()
                    loss_c2.backward()
                    torch.nn.utils.clip_grad_value_(critic2.parameters(), 1.0)
                    critic2_opt.step()

                    ep_td_losses.append((loss_c1.item() + loss_c2.item()) * 0.5)
                    ep_q_pred_avgs.append(q1_pred.detach().mean().item())
                    ep_target_q_avgs.append(target_q.detach().mean().item())

                    policy_update_ct += 1
                    if policy_update_ct % cfg.policy_delay == 0:
                        actor_loss = -critic1(obs_bt, actor(obs_bt)).mean()
                        actor_opt.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_value_(actor.parameters(), 1.0)
                        actor_opt.step()

                        _soft_update(actor,   target_actor,   cfg.tau)
                        _soft_update(critic1, target_critic1, cfg.tau)
                        _soft_update(critic2, target_critic2, cfg.tau)

            if is_done:
                break

        ep_eval_reward_avg = float('nan')
        if episode_ct % cfg.eval_interval == 0:
            ep_eval_reward_avg = _evaluate_policy(actor, cfg.eval_episodes, cfg.max_steps_per_ep)

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
            'actor':         actor.state_dict(),
            'target_actor':  target_actor.state_dict(),
            'critic1':       critic1.state_dict(),
            'target_critic1': target_critic1.state_dict(),
            'critic2':       critic2.state_dict(),
            'target_critic2': target_critic2.state_dict(),
            'res_net':       res_net.state_dict(),
            'uncert_model':  uncert_model.state_dict(),
            'actor_opt':     actor_opt.state_dict(),
            'critic1_opt':   critic1_opt.state_dict(),
            'critic2_opt':   critic2_opt.state_dict(),
            'res_net_opt':   res_net_opt.state_dict(),
            'rbf_opt':       rbf_opt.state_dict(),
            'episode':       episode_ct,
            'total_steps':   total_step_ct,
            'policy_update_ct':           policy_update_ct,
            'best_avg_score':             best_avg_score,
            'model_trained_at_least_once': model_trained_at_least_once,
            'rewards': episode_cumulative_reward_vector,
            'steps':   episode_step_vector,
            'real_buffer': _buf_state(real_buffer),
        }, resume_path)

        TARGET_SCORE = 2000.0   # Hopper-v5 목표 점수
        if len(episode_cumulative_reward_vector) >= 10:
            current_avg = np.mean(episode_cumulative_reward_vector[-10:])
            if current_avg > best_avg_score and current_avg >= TARGET_SCORE:
                best_avg_score = current_avg
                torch.save({
                    'actor'        : actor.state_dict(),
                    'critic1'      : critic1.state_dict(),
                    'critic2'      : critic2.state_dict(),
                    'res_net'      : res_net.state_dict(),
                    'uncert_model' : uncert_model.state_dict(),
                    'total_steps'  : total_step_ct,
                    'episode'      : episode_ct,
                }, checkpoint_path)
                print(f'[Seed {run_idx}] New best! Avg {current_avg:.1f} '
                      f'at ep {episode_ct} / step {total_step_ct}')

    env.close()
    return episode_cumulative_reward_vector, episode_step_vector, episode_metrics_list