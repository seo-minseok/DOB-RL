"""
trainer.py — DOB-MBRL BipedalWalker 메인 루프 (TD3 + resume 지원)
DQN (discrete) → TD3 (Twin Delayed DDPG, continuous action) 전환.
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
from .model_learning import train_residual_dx_model_dob, train_uncertainty_rbf, evaluate_rbf_calibration, train_contact_net
from .rollout import generate_samples_dob, sample_mixed_minibatch
from ..models import ActorNetwork, QNetwork, ResidualDxNet, NormalizedRBFModel, ContactNet
from ..utils.buffer import ReplayBufferDOB
from ..dynamics import (
    default_bipedalwalker_params, step_nominal_bipedalwalker,
    FPINV, F_MAT, DOB_DIM, OBS_DIM, ACT_DIM, OBS_DIM_NAMES,
)
from ..envs.bipedalwalker_utils import (
    make_bipedalwalker_env, reset_env, step_env,
)


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    with torch.no_grad():
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def train_DOB_core(run_idx: int,
                   num_episodes: int,
                   result_queue=None,
                   checkpoint_dir: str = '.',
                   resume: bool = False,
                   cfg: DOBMBRLConfig = None,
                   results_dir: str = None):
    """
    Args
    ----
    run_idx        : random seed (1-based)
    num_episodes   : 총 학습 에피소드 수
    result_queue   : multiprocessing.Queue
    checkpoint_dir : 체크포인트 저장 디렉토리
    resume         : True이면 기존 체크포인트에서 재개
    cfg            : DOBMBRLConfig 인스턴스 (None이면 기본값 사용)

    Returns
    -------
    episode_cumulative_reward_vector : list[float]
    episode_step_vector              : list[int]  (누적 총 스텝)
    """
    if cfg is None:
        cfg = DOBMBRLConfig()

    np.random.seed(run_idx)
    torch.manual_seed(run_idx)

    env              = make_bipedalwalker_env()
    num_observations = OBS_DIM   # 14
    num_act_features = ACT_DIM   # 4

    p_nom       = default_bipedalwalker_params()
    use_nominal = True
    use_dob     = True

    # --- TD3 Models ---
    actor         = ActorNetwork(num_observations, num_act_features)
    target_actor  = deepcopy(actor)
    actor_opt     = optim.Adam(actor.parameters(), lr=cfg.lr_actor)

    critic1       = QNetwork(num_observations, num_act_features)
    target_critic1 = deepcopy(critic1)
    critic1_opt   = optim.Adam(critic1.parameters(), lr=cfg.lr_critic)

    critic2       = QNetwork(num_observations, num_act_features)
    target_critic2 = deepcopy(critic2)
    critic2_opt   = optim.Adam(critic2.parameters(), lr=cfg.lr_critic)

    # 타깃 네트워크는 학습 안 함
    for p in target_actor.parameters():  p.requires_grad_(False)
    for p in target_critic1.parameters(): p.requires_grad_(False)
    for p in target_critic2.parameters(): p.requires_grad_(False)

    res_net     = ResidualDxNet(num_observations, num_act_features, hidden=64)
    res_net_opt = optim.SGD(res_net.parameters(), lr=cfg.lr_residual, momentum=0.9)

    uncert_model = NormalizedRBFModel(cfg.num_rbf_centers, cfg.rbf_width, cfg.rbf_initial_value)
    rbf_opt      = optim.SGD(uncert_model.parameters(), lr=cfg.lr_rbf, momentum=0.9)

    contact_net     = ContactNet(num_observations, num_act_features, hidden=64)
    contact_net_opt = optim.Adam(contact_net.parameters(), lr=cfg.lr_residual)

    # --- Buffers ---
    real_buffer  = ReplayBufferDOB(cfg.buffer_size, num_observations, num_act_features)
    model_buffer = ReplayBufferDOB(cfg.buffer_size, num_observations, num_act_features)

    sample_gen_options = {
        'max_horizon_length'            : cfg.max_horizon_length,
        'uncertainty_threshold'         : cfg.uncertainty_threshold,
        'num_generate_sample_iteration' : cfg.num_generate_sample_iteration,
        'mini_batch_size'               : cfg.mini_batch_size,
        'epsilon_min_model'             : cfg.epsilon_min_model,
    }

    episode_cumulative_reward_vector = []
    episode_step_vector              = []
    total_step_ct                    = 0
    total_grad_steps                 = 0
    model_trained_at_least_once      = False
    best_avg_score                   = -float('inf')
    start_episode                    = 1

    # --- Resume ---
    checkpoint_path = os.path.join(checkpoint_dir, f'Champion_Seed{run_idx}_BestModel.pt')
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        actor.load_state_dict(ckpt['actor'])
        target_actor.load_state_dict(ckpt['actor'])
        critic1.load_state_dict(ckpt['critic1'])
        critic2.load_state_dict(ckpt['critic2'])
        target_critic1.load_state_dict(ckpt['critic1'])
        target_critic2.load_state_dict(ckpt['critic2'])
        res_net.load_state_dict(ckpt['res_net'])
        uncert_model.load_state_dict(ckpt['uncert_model'])
        if 'contact_net' in ckpt:
            contact_net.load_state_dict(ckpt['contact_net'])
        start_episode    = ckpt['episode'] + 1
        total_step_ct    = ckpt['total_steps']
        total_grad_steps = ckpt.get('total_grad_steps', 0)
        print(f'[Seed {run_idx}] Resumed from episode {ckpt["episode"]} '
              f'(total_steps={total_step_ct})')

    # --- Main Training Loop ---
    for episode_ct in range(start_episode, num_episodes + 1):

        ep_res_net_loss        = float('nan')
        ep_contact_net_loss    = float('nan')
        ep_rbf_loss            = float('nan')
        ep_buffer_uncert_avg   = float('nan')
        ep_sampled_uncert_avg  = float('nan')
        ep_rollout_uncert_avg  = float('nan')
        ep_rollout_pass_rate   = float('nan')
        ep_rollout_avg_horizon = float('nan')
        ep_rbf_calib_ratio     = float('nan')
        ep_rbf_calib_corr      = float('nan')

        # [Phase 1] Model Training & Rollout
        if real_buffer.length > cfg.mini_batch_size and total_step_ct > cfg.warm_start_samples:
            if cfg.real_ratio < 1.0:
                valid_len = real_buffer.length
                ep_buffer_uncert_avg = float(
                    np.linalg.norm(real_buffer.uncertainty[:valid_len], axis=1).mean()
                )
                ep_res_net_loss, ep_sampled_uncert_avg = train_residual_dx_model_dob(
                    res_net, res_net_opt, real_buffer,
                    cfg.mini_batch_size, cfg.num_epochs
                )
                ep_contact_net_loss = train_contact_net(
                    contact_net, contact_net_opt, real_buffer,
                    cfg.mini_batch_size, cfg.num_epochs
                )
                ep_rbf_loss = train_uncertainty_rbf(
                    uncert_model, rbf_opt, real_buffer, res_net,
                    cfg.mini_batch_size, 5
                )
                ep_rbf_calib_ratio, ep_rbf_calib_corr = evaluate_rbf_calibration(
                    uncert_model, real_buffer
                )
                model_trained_at_least_once = True
                model_buffer.reset()   # stale synthetic transition 제거
                (model_buffer,
                 ep_rollout_uncert_avg,
                 ep_rollout_pass_rate,
                 ep_rollout_avg_horizon) = generate_samples_dob(
                    real_buffer, model_buffer, res_net, uncert_model,
                    actor, cfg.epsilon_min_model, sample_gen_options,
                    p_nom, use_nominal, contact_net
                )

        # [Phase 2] Episode Reset
        obs            = np.array(reset_env(env), dtype=np.float32)
        obs_t          = torch.tensor(obs).unsqueeze(0)
        episode_reward = 0.0
        dhat           = np.zeros(DOB_DIM, dtype=np.float32)   # 에피소드마다 리셋
        ep_nominal_errors   = []
        ep_residual_errors  = []
        ep_dhat_norms       = []
        ep_uncertainty_mags = []
        ep_td_losses        = []
        ep_nom_err_per_dim  = []   # list of (OBS_DIM,) abs error arrays
        ep_res_err_per_dim  = []   # list of (OBS_DIM,) abs error arrays
        ep_contact_errs     = []   # list of (2,) abs error: [left_contact, right_contact]

        # [Phase 3] Environment Interaction
        for step_ct in range(1, cfg.max_steps_per_ep + 1):
            total_step_ct += 1

            # --- Action Selection (TD3 탐색) ---
            if total_step_ct <= cfg.warm_start_samples:
                action = np.random.uniform(-1.0, 1.0, size=num_act_features).astype(np.float32)
            else:
                with torch.no_grad():
                    action = actor(obs_t).cpu().numpy().flatten()
                noise  = np.random.normal(0.0, cfg.expl_noise, size=action.shape).astype(np.float32)
                action = np.clip(action + noise, -1.0, 1.0)

            # --- Env Step ---
            next_obs_raw, env_reward, is_done, _ = step_env(env, action)
            next_obs = np.array(next_obs_raw, dtype=np.float32)
            reward   = float(env_reward)

            # --- DOB Online Update ---
            if use_nominal:
                x_nom_next = step_nominal_bipedalwalker(
                    obs.reshape(1, -1),
                    action.reshape(1, -1),
                    p_nom
                ).flatten()
                dx_nom = x_nom_next - obs
            else:
                dx_nom = np.zeros_like(obs)

            dx_real = next_obs - obs

            with torch.no_grad():
                inp_res = torch.tensor(
                    np.concatenate([obs, action], dtype=np.float32)
                ).unsqueeze(0)
                dx_res = res_net(inp_res).cpu().numpy().flatten()   # (7,)

            e = dx_real - dx_nom   # (14,)
            if use_dob:
                dhat = cfg.dob_w * dx_res + (1.0 - cfg.dob_w) * (FPINV @ e)
            else:
                dhat = np.zeros(DOB_DIM, dtype=np.float32)

            uncertainty = FPINV @ e - dx_res   # (7,)

            ep_nominal_errors.append(float(np.linalg.norm(e)))
            ep_residual_errors.append(float(np.linalg.norm(e - F_MAT @ dx_res)))
            ep_dhat_norms.append(float(np.linalg.norm(dhat)))
            ep_uncertainty_mags.append(float(np.linalg.norm(uncertainty)))
            ep_nom_err_per_dim.append(np.abs(e).astype(np.float32))
            ep_res_err_per_dim.append(np.abs(e - F_MAT @ dx_res).astype(np.float32))

            with torch.no_grad():
                contact_pred = contact_net(inp_res).cpu().numpy().flatten()   # (2,)
            ep_contact_errs.append(np.abs(contact_pred - next_obs[[8, 13]]).astype(np.float32))

            real_buffer.store(
                obs         = obs,
                act         = action,
                next_obs    = next_obs,
                rew         = np.float32(reward),
                done        = is_done,
                dhat        = dhat,
                dx_nom      = dx_nom,
                uncertainty = uncertainty,
            )

            episode_reward += reward
            obs   = next_obs
            obs_t = torch.tensor(obs).unsqueeze(0)

            # --- [Phase 4] TD3 Agent Update ---
            if (step_ct % cfg.update_interval == 0) and (total_step_ct > cfg.warm_start_samples):
                total_updates = cfg.num_gradient_steps * cfg.update_interval

                for _ in range(total_updates):
                    if real_buffer.length < cfg.mini_batch_size:
                        break
                    total_grad_steps += 1

                    s_obs, s_act, s_nxt, s_rew, s_done = sample_mixed_minibatch(
                        model_trained_at_least_once, cfg.real_ratio,
                        cfg.mini_batch_size, real_buffer, model_buffer
                    )

                    obs_bt  = torch.tensor(s_obs)
                    act_bt  = torch.tensor(s_act)
                    nxt_bt  = torch.tensor(s_nxt)
                    rew_bt  = torch.tensor(s_rew)
                    done_bt = torch.tensor(s_done, dtype=torch.bool)

                    # --- Critic 타깃 계산 (target policy noise + double Q) ---
                    with torch.no_grad():
                        noise_t = torch.clamp(
                            torch.randn_like(act_bt) * cfg.policy_noise,
                            -cfg.noise_clip, cfg.noise_clip
                        )
                        tgt_act = torch.clamp(target_actor(nxt_bt) + noise_t, -1.0, 1.0)
                        q1_next = target_critic1(nxt_bt, tgt_act)
                        q2_next = target_critic2(nxt_bt, tgt_act)
                        target_q = rew_bt + cfg.discount_factor * torch.min(q1_next, q2_next)
                        target_q[done_bt] = rew_bt[done_bt]

                    # --- Critic 1 업데이트 ---
                    q1_pred = critic1(obs_bt, act_bt)
                    loss_c1 = nn.functional.mse_loss(q1_pred, target_q)
                    critic1_opt.zero_grad()
                    loss_c1.backward()
                    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
                    critic1_opt.step()

                    # --- Critic 2 업데이트 ---
                    q2_pred = critic2(obs_bt, act_bt)
                    loss_c2 = nn.functional.mse_loss(q2_pred, target_q)
                    critic2_opt.zero_grad()
                    loss_c2.backward()
                    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
                    critic2_opt.step()

                    ep_td_losses.append((loss_c1.item() + loss_c2.item()) / 2.0)

                    # --- Actor 업데이트 (policy_delay) ---
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

        _nan_dim = [float('nan')] * OBS_DIM
        nom_err_per_dim    = np.mean(ep_nom_err_per_dim, axis=0).tolist() if ep_nom_err_per_dim else _nan_dim
        res_err_per_dim    = np.mean(ep_res_err_per_dim, axis=0).tolist() if ep_res_err_per_dim else _nan_dim
        contact_err_avg    = np.mean(ep_contact_errs,    axis=0).tolist() if ep_contact_errs    else [float('nan'), float('nan')]

        ep_metrics = {
            'nominal_error_avg':    float(np.mean(ep_nominal_errors))   if ep_nominal_errors   else float('nan'),
            'residual_error_avg':   float(np.mean(ep_residual_errors))  if ep_residual_errors  else float('nan'),
            'dhat_norm_avg':        float(np.mean(ep_dhat_norms))       if ep_dhat_norms       else float('nan'),
            'uncertainty_avg':      float(np.mean(ep_uncertainty_mags)) if ep_uncertainty_mags else float('nan'),
            'res_net_loss':         ep_res_net_loss,
            'contact_net_loss':     ep_contact_net_loss,
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
            'contact_err_left':     contact_err_avg[0],
            'contact_err_right':    contact_err_avg[1],
        }

        episode_cumulative_reward_vector.append(episode_reward)
        episode_step_vector.append(total_step_ct)

        # --- Per-episode incremental CSV save ---
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            csv_path     = os.path.join(results_dir, f'seed_{run_idx}_progress.csv')
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'w' if write_header else 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'seed', 'episode', 'total_steps', 'reward',
                        'nominal_error_avg', 'residual_error_avg', 'dhat_norm_avg',
                        'uncertainty_avg', 'res_net_loss', 'contact_net_loss',
                        'rbf_loss', 'td_loss_avg',
                        'episode_length', 'expl_noise', 'buffer_uncert_avg',
                        'sampled_uncert_avg', 'rollout_uncert_avg',
                        'rollout_pass_rate', 'rollout_avg_horizon',
                        'rbf_calib_ratio', 'rbf_calib_corr',
                        *[f'nom_err_{n}' for n in OBS_DIM_NAMES],
                        *[f'res_err_{n}' for n in OBS_DIM_NAMES],
                        'contact_err_left', 'contact_err_right',
                    ])
                writer.writerow([
                    run_idx, episode_ct, total_step_ct, episode_reward,
                    ep_metrics['nominal_error_avg'], ep_metrics['residual_error_avg'],
                    ep_metrics['dhat_norm_avg'], ep_metrics['uncertainty_avg'],
                    ep_metrics['res_net_loss'], ep_metrics['contact_net_loss'],
                    ep_metrics['rbf_loss'], ep_metrics['td_loss_avg'],
                    ep_metrics['episode_length'],
                    ep_metrics['expl_noise'], ep_metrics['buffer_uncert_avg'],
                    ep_metrics['sampled_uncert_avg'], ep_metrics['rollout_uncert_avg'],
                    ep_metrics['rollout_pass_rate'], ep_metrics['rollout_avg_horizon'],
                    ep_metrics['rbf_calib_ratio'], ep_metrics['rbf_calib_corr'],
                    *nom_err_per_dim,
                    *res_err_per_dim,
                    ep_metrics['contact_err_left'],
                    ep_metrics['contact_err_right'],
                ])

        if result_queue is not None:
            result_queue.put({
                'run_idx': run_idx,
                'ep_idx' : episode_ct,
                'reward' : episode_reward,
                'step'   : total_step_ct,
                **ep_metrics,
            })

        # --- Checkpoint (10-ep avg 기준, 새 best마다 저장) ---
        if len(episode_cumulative_reward_vector) >= 10:
            current_avg = np.mean(episode_cumulative_reward_vector[-10:])
            if current_avg > best_avg_score:
                best_avg_score = current_avg
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'actor'           : actor.state_dict(),
                    'critic1'         : critic1.state_dict(),
                    'critic2'         : critic2.state_dict(),
                    'res_net'         : res_net.state_dict(),
                    'uncert_model'    : uncert_model.state_dict(),
                    'contact_net'     : contact_net.state_dict(),
                    'total_steps'     : total_step_ct,
                    'total_grad_steps': total_grad_steps,
                    'episode'         : episode_ct,
                    'config'          : dataclasses.asdict(cfg),
                }, checkpoint_path)
                print(f'[Seed {run_idx}] New best! Avg {current_avg:.1f} '
                      f'at ep {episode_ct} / step {total_step_ct}')

    env.close()
    return episode_cumulative_reward_vector, episode_step_vector
