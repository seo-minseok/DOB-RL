"""
trainer.py — train_DOB_core 메인 루프 래핑 + resume 지원
MATLAB: train_DOB_core(runIdx, numEpisodes, dataQueue)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from .config import DOBMBRLConfig
from .model_learning import train_residual_dx_model_dob, train_uncertainty_rbf
from .rollout import generate_samples_dob, sample_mixed_minibatch
from ..models import QNetwork, ResidualDxNet, NormalizedRBFModel
from ..utils.buffer import ReplayBufferDOB
from ..dynamics import (
    default_cartpole_params, step_nominal_cartpole,
    ACT_ELEMENTS, FPINV,
)
from ..envs.cartpole_utils import (
    make_cartpole_env, reset_env, step_env, reward_is_done_function,
)


def train_DOB_core(run_idx: int,
                   num_episodes: int,
                   result_queue=None,
                   checkpoint_dir: str = '.',
                   resume: bool = False,
                   cfg: DOBMBRLConfig = None):
    """
    Args
    ----
    run_idx        : random seed (1-based, MATLAB runIdx 동일)
    num_episodes   : 총 학습 에피소드 수
    result_queue   : multiprocessing.Queue (MATLAB: dataQueue)
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

    env              = make_cartpole_env()
    num_observations = 4
    num_actions      = 2
    num_act_features = 1

    p_nom       = default_cartpole_params()
    use_nominal = True
    use_dob     = True
    dhat        = np.zeros(2, dtype=np.float32)

    # --- Models ---
    q_network      = QNetwork(num_observations, num_actions)
    target_network = deepcopy(q_network)
    critic_opt     = optim.Adam(q_network.parameters(), lr=cfg.lr_critic, weight_decay=0.0)

    res_net     = ResidualDxNet(num_observations, num_act_features, hidden=32)
    res_net_opt = optim.SGD(res_net.parameters(), lr=cfg.lr_residual, momentum=0.9)

    uncert_model = NormalizedRBFModel(cfg.num_rbf_centers, cfg.rbf_width, cfg.rbf_initial_value)
    rbf_opt      = optim.SGD(uncert_model.parameters(), lr=cfg.lr_rbf, momentum=0.9)

    # --- Buffers ---
    real_buffer  = ReplayBufferDOB(cfg.buffer_size, num_observations, num_act_features)
    model_buffer = ReplayBufferDOB(cfg.buffer_size, num_observations, num_act_features)

    sample_gen_options = {
        'max_horizon_length'            : cfg.max_horizon_length,
        'uncertainty_threshold'         : cfg.uncertainty_threshold,
        'num_generate_sample_iteration' : cfg.num_generate_sample_iteration,
        'mini_batch_size'               : cfg.mini_batch_size,
        'num_observations'              : num_observations,
        'epsilon_min_model'             : cfg.epsilon_min_model,
    }

    episode_cumulative_reward_vector = []
    episode_step_vector              = []
    total_step_ct                    = 0
    model_trained_at_least_once      = False
    best_avg_score                   = -float('inf')
    epsilon                          = cfg.epsilon
    start_episode                    = 1

    act_elements_t = torch.tensor(ACT_ELEMENTS)

    # --- Resume ---
    checkpoint_path = os.path.join(checkpoint_dir, f'Champion_Seed{run_idx}_BestModel.pt')
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        q_network.load_state_dict(ckpt['q_network'])
        target_network.load_state_dict(ckpt['q_network'])
        res_net.load_state_dict(ckpt['res_net'])
        uncert_model.load_state_dict(ckpt['uncert_model'])
        start_episode = ckpt['episode'] + 1
        total_step_ct = ckpt['total_steps']
        epsilon       = ckpt.get('epsilon', cfg.epsilon_min)
        print(f'[Seed {run_idx}] Resumed from episode {ckpt["episode"]} '
              f'(total_steps={total_step_ct})')

    # --- Main Training Loop ---
    for episode_ct in range(start_episode, num_episodes + 1):

        # [Phase 1] Model Training & Rollout
        if real_buffer.length > cfg.mini_batch_size and total_step_ct > cfg.warm_start_samples:
            if cfg.real_ratio < 1.0:
                train_residual_dx_model_dob(
                    res_net, res_net_opt, real_buffer,
                    cfg.mini_batch_size, cfg.num_epochs,
                    use_uncertainty_sampling=cfg.use_uncertainty_sampling,
                )
                train_uncertainty_rbf(
                    uncert_model, rbf_opt, real_buffer, res_net,
                    cfg.mini_batch_size, 5
                )
                model_trained_at_least_once = True
                model_buffer = generate_samples_dob(
                    real_buffer, model_buffer, res_net, uncert_model,
                    q_network, epsilon, sample_gen_options, p_nom, use_nominal
                )

        # [Phase 2] Episode Reset
        obs            = np.array(reset_env(env), dtype=np.float32)
        obs_t          = torch.tensor(obs).unsqueeze(0)
        episode_reward = 0.0
        dhat           = np.zeros(2, dtype=np.float32)   # 에피소드마다 리셋

        # [Phase 3] Environment Interaction
        for step_ct in range(1, cfg.max_steps_per_ep + 1):
            total_step_ct += 1

            if np.random.rand() < epsilon:
                action_idx = np.random.randint(0, num_actions)
            else:
                with torch.no_grad():
                    action_idx = int(q_network(obs_t).argmax(dim=1).item())

            action_force = float(ACT_ELEMENTS[action_idx])
            gym_action   = action_idx

            if total_step_ct > cfg.warm_start_samples:
                epsilon = max(epsilon * (1.0 - cfg.epsilon_decay), cfg.epsilon_min)

            next_obs_raw, _, is_done, _ = step_env(env, gym_action)
            next_obs = np.array(next_obs_raw, dtype=np.float32)

            # DOB online update
            dx_real = next_obs - obs
            if use_nominal:
                x_nom_next = step_nominal_cartpole(
                    obs.reshape(1, -1),
                    np.array([[action_force]], dtype=np.float32),
                    p_nom
                ).flatten()
                dx_nom = x_nom_next - obs
            else:
                dx_nom = np.zeros_like(obs)

            with torch.no_grad():
                inp_res = torch.tensor(
                    np.concatenate([obs, [action_force]], dtype=np.float32)
                ).unsqueeze(0)
                dx_res = res_net(inp_res).cpu().numpy().flatten()   # (2,)

            e = dx_real - dx_nom
            if use_dob:
                dhat = cfg.dob_w * dx_res + (1.0 - cfg.dob_w) * (FPINV @ e)
            else:
                dhat = np.zeros(2, dtype=np.float32)

            uncertainty = FPINV @ e - dx_res

            reward_arr, _ = reward_is_done_function(next_obs.reshape(1, -1))
            reward = float(reward_arr[0])

            real_buffer.store(
                obs         = obs,
                act         = np.array([action_force], dtype=np.float32),
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

            # [Phase 4] Agent Update (UTD-10)
            if (step_ct % cfg.update_interval == 0) and (total_step_ct > cfg.warm_start_samples):
                total_updates = cfg.num_gradient_steps * cfg.update_interval

                for _ in range(total_updates):
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
                        max_next_q = target_network(nxt_bt).max(dim=1).values

                    target_q          = rew_bt + cfg.discount_factor * max_next_q
                    target_q[done_bt] = rew_bt[done_bt]

                    act_mask = (act_elements_t.unsqueeze(0) == act_bt).float()
                    q_pred   = (q_network(obs_bt) * act_mask).sum(dim=1)
                    loss     = nn.functional.mse_loss(q_pred, target_q)

                    critic_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(q_network.parameters(), 1.0)
                    critic_opt.step()

                    with torch.no_grad():
                        for p, tp in zip(q_network.parameters(), target_network.parameters()):
                            tp.data.copy_(cfg.tau * p.data + (1.0 - cfg.tau) * tp.data)

            if is_done:
                break

        episode_cumulative_reward_vector.append(episode_reward)
        episode_step_vector.append(total_step_ct)

        if result_queue is not None:
            result_queue.put({
                'run_idx': run_idx,
                'ep_idx' : episode_ct,
                'reward' : episode_reward,
                'step'   : total_step_ct,
            })

        if len(episode_cumulative_reward_vector) >= 10:
            current_avg = np.mean(episode_cumulative_reward_vector[-10:])
            if current_avg > best_avg_score and current_avg >= 480:
                best_avg_score = current_avg
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'q_network'    : q_network.state_dict(),
                    'res_net'      : res_net.state_dict(),
                    'uncert_model' : uncert_model.state_dict(),
                    'total_steps'  : total_step_ct,
                    'episode'      : episode_ct,
                    'epsilon'      : epsilon,
                }, checkpoint_path)
                print(f'[Seed {run_idx}] New best! Avg {current_avg:.1f} '
                      f'at ep {episode_ct} / step {total_step_ct}')

    env.close()
    return episode_cumulative_reward_vector, episode_step_vector
