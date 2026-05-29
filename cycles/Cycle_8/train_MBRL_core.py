"""
train_MBRL_core.py  (Cycle 1 local copy)

base/original/train_MBRL_core.py 와 동일하나 td_loss 수집 추가.
변경점:
  - episode 단위 평균 Q-network td_loss 수집 → episode_td_loss_vector
  - 반환값: (rewards, steps, td_losses)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import os
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.getenv("MBRL_RESULTS_DIR", str(PROJECT_ROOT / "checkpoints" / "mbrl")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OBS_MIN      = np.array([-2.4, -4.0, -0.209, -3.5], dtype=np.float32)
OBS_MAX      = np.array([ 2.4,  4.0,  0.209,  3.5], dtype=np.float32)
ACT_ELEMENTS = np.array([-10.0, 10.0], dtype=np.float32)
ACT_MIN      = -10.0
ACT_MAX      =  10.0


class QNetwork(nn.Module):
    def __init__(self, num_observations: int = 4, num_actions: int = 2):
        super().__init__()
        self.register_buffer('obs_min', torch.tensor(OBS_MIN))
        self.register_buffer('obs_max', torch.tensor(OBS_MAX))
        self.fc1 = nn.Linear(num_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_norm = 2.0 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1.0
        x = torch.relu(self.fc1(obs_norm))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TransitionNetwork(nn.Module):
    def __init__(self, num_observations: int = 4, num_act_features: int = 1):
        super().__init__()
        hidden1 = 32
        hidden2 = 32
        self.register_buffer('obs_min', torch.tensor(OBS_MIN))
        self.register_buffer('obs_max', torch.tensor(OBS_MAX))
        self.act_min = ACT_MIN
        self.act_max = ACT_MAX
        self.fc1 = nn.Linear(num_observations + num_act_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_observations)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        obs_norm = 2.0 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1.0
        act_norm = 2.0 * (act - self.act_min) / (self.act_max - self.act_min) - 1.0
        x = torch.cat([obs_norm, act_norm], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, buffer_size: int, num_obs: int = 4, num_act: int = 1):
        self.size   = buffer_size
        self.index  = 0
        self.length = 0
        self.obs      = np.zeros((buffer_size, num_obs), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, num_obs), dtype=np.float32)
        self.act      = np.zeros((buffer_size, num_act), dtype=np.float32)
        self.rew      = np.zeros(buffer_size,            dtype=np.float32)
        self.done     = np.zeros(buffer_size,            dtype=bool)

    def store(self, obs, act, next_obs, rew, done):
        idx = self.index
        self.obs[idx]      = obs
        self.act[idx]      = act
        self.next_obs[idx] = next_obs
        self.rew[idx]      = rew
        self.done[idx]     = done
        self.index  = (self.index + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def store_batch(self, obs_b, act_b, next_obs_b, rew_b, done_b):
        n       = len(obs_b)
        indices = np.arange(self.index, self.index + n) % self.size
        self.obs[indices]      = obs_b
        self.act[indices]      = act_b
        self.next_obs[indices] = next_obs_b
        self.rew[indices]      = rew_b
        self.done[indices]     = done_b
        self.index  = (self.index + n) % self.size
        self.length = min(self.length + n, self.size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.length, size=batch_size)
        return (self.obs[idx], self.act[idx], self.next_obs[idx],
                self.rew[idx], self.done[idx])


def reward_is_done_function(next_obs):
    theta_threshold = 12.0 * np.pi / 180.0
    x_threshold     = 2.4
    if isinstance(next_obs, torch.Tensor):
        next_obs = next_obs.detach().cpu().numpy()
    next_obs = np.asarray(next_obs, dtype=np.float32)
    if next_obs.ndim == 1:
        next_obs = next_obs.reshape(1, -1)
    x     = next_obs[:, 0]
    theta = next_obs[:, 2]
    is_done     = (np.abs(x) > x_threshold) | (np.abs(theta) > theta_threshold)
    r_angle     = 1.0 - (np.abs(theta) / theta_threshold) ** 2
    r_pos       = 1.0 - (np.abs(x)     / x_threshold)     ** 2
    shaped_rew  = 0.4 * 1.0 + 0.4 * r_angle + 0.2 * r_pos
    reward = np.where(is_done, np.float32(-10.0), shaped_rew.astype(np.float32))
    return reward, is_done


def train_transition_model(model: TransitionNetwork,
                           optimizer: optim.SGD,
                           real_buffer: ReplayBuffer,
                           mini_batch_size: int,
                           num_epochs: int) -> float:
    model.train()
    loss_val = 0.0
    for _ in range(num_epochs):
        perm           = np.random.permutation(real_buffer.length)
        num_iterations = real_buffer.length // mini_batch_size
        for it in range(num_iterations):
            idx       = perm[it * mini_batch_size:(it + 1) * mini_batch_size]
            obs_t     = torch.tensor(real_buffer.obs[idx])
            act_t     = torch.tensor(real_buffer.act[idx])
            next_t    = torch.tensor(real_buffer.next_obs[idx])
            optimizer.zero_grad()
            dx_pred = model(obs_t, act_t)
            dx_true = next_t - obs_t
            loss    = nn.functional.mse_loss(dx_pred, dx_true)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
    return loss_val


def generate_samples(real_buffer, model_buffer, transition_models,
                     q_network, epsilon, options, episode_ct):
    num_models         = len(transition_models)
    min_horizon        = 1
    max_horizon        = options['horizon_length']
    start_ep           = 20
    end_ep             = 200
    num_iterations     = options['num_generate_sample_iteration']
    mini_batch_size    = options['mini_batch_size']
    epsilon_min_model  = options['epsilon_min_model']

    if episode_ct <= start_ep:
        horizon_length = min_horizon
    elif episode_ct >= end_ep:
        horizon_length = max_horizon
    else:
        slope          = (max_horizon - min_horizon) / (end_ep - start_ep)
        horizon_length = int(min_horizon + slope * (episode_ct - start_ep))

    act_elements_t = torch.tensor(ACT_ELEMENTS)

    for _ in range(num_iterations):
        predicted_next_obs = [None] * num_models

        for h in range(horizon_length):
            model_order = np.random.permutation(num_models)

            for mi, model_id in enumerate(model_order):
                if real_buffer.length < mini_batch_size:
                    continue

                model = transition_models[model_id]
                model.eval()

                if h == 0:
                    idx      = np.random.choice(real_buffer.length,
                                                mini_batch_size, replace=False)
                    curr_obs = torch.tensor(real_buffer.obs[idx])
                else:
                    curr_obs = predicted_next_obs[mi]

                with torch.no_grad():
                    q_vals = q_network(curr_obs)
                _, action_indices = q_vals.max(dim=1)
                sampled_force = act_elements_t[action_indices].unsqueeze(1)

                rand_idx     = np.random.randint(0, 2, mini_batch_size)
                random_force = act_elements_t[rand_idx].unsqueeze(1)

                eps_floor = max(epsilon, epsilon_min_model)
                eps_mask  = torch.rand(mini_batch_size) < eps_floor
                sampled_force = sampled_force.clone()
                sampled_force[eps_mask] = random_force[eps_mask]

                with torch.no_grad():
                    dx            = model(curr_obs, sampled_force)
                pred_next_obs = curr_obs + dx
                predicted_next_obs[mi] = pred_next_obs

                pred_next_np   = pred_next_obs.numpy()
                sampled_reward, sampled_done = reward_is_done_function(pred_next_np)

                model_buffer.store_batch(
                    obs_b      = curr_obs.numpy(),
                    act_b      = sampled_force.numpy(),
                    next_obs_b = pred_next_np,
                    rew_b      = sampled_reward,
                    done_b     = sampled_done,
                )

    return model_buffer


def sample_mixed_minibatch(model_trained, real_ratio, mini_batch_size,
                           real_buffer, model_buffer):
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


def train_MBRL_core(run_idx: int, num_episodes: int):
    """
    Returns
    -------
    episode_cumulative_reward_vector : list[float]
    episode_step_vector              : list[int]
    episode_td_loss_vector           : list[float]  ← 추가: episode 평균 Q-network td_loss
    """
    np.random.seed(run_idx)
    torch.manual_seed(run_idx)

    env = gym.make('CartPole-v1')
    num_observations  = 4
    num_actions       = 2
    num_act_features  = 1

    q_network      = QNetwork(num_observations, num_actions)
    target_network = deepcopy(q_network)
    critic_optimizer = optim.Adam(q_network.parameters(), lr=1e-3, weight_decay=0.0)

    num_models = 3
    transition_models = [
        TransitionNetwork(num_observations, num_act_features)
        for _ in range(num_models)
    ]
    model_optimizers = [
        optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)
        for m in transition_models
    ]
    num_epochs = 5

    buffer_size         = int(1e5)
    max_steps_per_ep    = 500
    discount_factor     = 0.99
    mini_batch_size     = 256
    warm_start_samples  = 200
    epsilon             = 1.0
    epsilon_min         = 0.01
    epsilon_decay       = 0.005
    real_ratio          = 0.2
    num_gradient_steps  = 2
    update_interval     = 10
    epsilon_min_model   = 0.1
    tau                 = 0.005

    sample_gen_options = {
        'horizon_length'                : 5,
        'num_generate_sample_iteration' : 20,
        'mini_batch_size'               : mini_batch_size,
        'num_observations'              : num_observations,
        'epsilon_min_model'             : epsilon_min_model,
    }

    real_buffer  = ReplayBuffer(buffer_size, num_observations, num_act_features)
    model_buffer = ReplayBuffer(buffer_size, num_observations, num_act_features)

    episode_cumulative_reward_vector = []
    episode_step_vector              = []
    episode_td_loss_vector           = []   # ← 추가
    total_step_ct                    = 0
    model_trained_at_least_once      = False
    best_avg_score                   = -float("inf")
    act_elements_t = torch.tensor(ACT_ELEMENTS)

    for episode_ct in range(1, num_episodes + 1):

        if real_buffer.length > mini_batch_size and total_step_ct > warm_start_samples:
            if real_ratio < 1.0:
                for model, opt in zip(transition_models, model_optimizers):
                    train_transition_model(model, opt, real_buffer,
                                          mini_batch_size, num_epochs)
                model_trained_at_least_once = True
                model_buffer = generate_samples(
                    real_buffer, model_buffer, transition_models,
                    q_network, epsilon, sample_gen_options, episode_ct
                )

        obs_raw, _ = env.reset()
        obs        = torch.tensor(obs_raw, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0.0
        ep_loss_sum    = 0.0   # ← 추가
        ep_loss_count  = 0     # ← 추가

        for step_ct in range(1, max_steps_per_ep + 1):
            total_step_ct += 1

            if np.random.rand() < epsilon:
                action_idx = np.random.randint(0, num_actions)
            else:
                with torch.no_grad():
                    q_vals = q_network(obs)
                action_idx = int(q_vals.argmax(dim=1).item())

            action_force = float(ACT_ELEMENTS[action_idx])
            gym_action   = action_idx

            if total_step_ct > warm_start_samples:
                epsilon = max(epsilon * (1.0 - epsilon_decay), epsilon_min)

            next_obs_raw, _, terminated, truncated, _ = env.step(gym_action)
            is_done = bool(terminated)

            reward_arr, _ = reward_is_done_function(
                np.array(next_obs_raw, dtype=np.float32).reshape(1, -1)
            )
            reward = float(reward_arr[0])
            next_obs = torch.tensor(next_obs_raw, dtype=torch.float32).unsqueeze(0)

            real_buffer.store(
                obs      = obs.numpy().flatten(),
                act      = np.array([action_force], dtype=np.float32),
                next_obs = next_obs.numpy().flatten(),
                rew      = np.float32(reward),
                done     = is_done,
            )

            episode_reward += reward
            obs = next_obs

            if (step_ct % update_interval == 0) and (total_step_ct > warm_start_samples):
                total_updates = num_gradient_steps * update_interval

                for _ in range(total_updates):
                    s_obs, s_act, s_nxt, s_rew, s_done = sample_mixed_minibatch(
                        model_trained_at_least_once, real_ratio,
                        mini_batch_size, real_buffer, model_buffer
                    )

                    obs_t      = torch.tensor(s_obs)
                    next_obs_t = torch.tensor(s_nxt)
                    act_t      = torch.tensor(s_act)
                    rew_t      = torch.tensor(s_rew)
                    done_t     = torch.tensor(s_done, dtype=torch.bool)

                    with torch.no_grad():
                        q_next     = target_network(next_obs_t)
                        max_next_q = q_next.max(dim=1).values

                    target_q            = rew_t + discount_factor * max_next_q
                    target_q[done_t]    = rew_t[done_t]

                    action_mask = (act_elements_t.unsqueeze(0) == act_t).float()
                    q_vals   = q_network(obs_t)
                    q_pred   = (q_vals * action_mask).sum(dim=1)

                    loss = nn.functional.mse_loss(q_pred, target_q)

                    critic_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(q_network.parameters(), 1.0)
                    critic_optimizer.step()

                    with torch.no_grad():
                        for p, tp in zip(q_network.parameters(),
                                         target_network.parameters()):
                            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

                    ep_loss_sum   += loss.item()   # ← 추가
                    ep_loss_count += 1             # ← 추가

            if is_done or truncated:
                break

        episode_cumulative_reward_vector.append(episode_reward)
        episode_step_vector.append(total_step_ct)
        # episode 평균 td_loss (업데이트 없으면 0)
        ep_td_loss = ep_loss_sum / ep_loss_count if ep_loss_count > 0 else 0.0
        episode_td_loss_vector.append(ep_td_loss)   # ← 추가

        if len(episode_cumulative_reward_vector) >= 10:
            current_avg = np.mean(episode_cumulative_reward_vector[-10:])
            if current_avg > best_avg_score and current_avg >= 480:
                best_avg_score = current_avg
                model_path = RESULTS_DIR / f"Champion_Seed{run_idx}_BestModel.pt"
                torch.save(
                    {
                        "q_network": q_network.state_dict(),
                        "transition_models": [m.state_dict() for m in transition_models],
                        "total_steps": total_step_ct,
                        "episode": episode_ct,
                    },
                    model_path,
                )
                print(
                    f"[Seed {run_idx}] New best! Avg {current_avg:.1f} "
                    f"at ep {episode_ct} / step {total_step_ct}"
                )

        if episode_ct % 50 == 0:
            print(
                f"[Run {run_idx}] Ep: {episode_ct}, "
                f"Score: {episode_reward:.1f}, "
                f"TotalSteps: {total_step_ct}, "
                f"Eps: {epsilon:.3f}, "
                f"TDLoss: {ep_td_loss:.4f}"
            )

    env.close()
    return episode_cumulative_reward_vector, episode_step_vector, episode_td_loss_vector
