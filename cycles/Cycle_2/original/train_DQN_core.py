"""
train_DQN_core.py

PyTorch + Gymnasium implementation of train_DQN_core.m
All network architectures, hyperparameters, and logic are identical to the MATLAB version.
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
RESULTS_DIR = Path(os.getenv("DQN_RESULTS_DIR", str(PROJECT_ROOT / "Python" / "DQN")))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Constants  (MATLAB: obsMin / obsMax / actInfo.Elements)
# ============================================================
OBS_MIN      = np.array([-2.4, -4.0, -0.209, -3.5], dtype=np.float32)
OBS_MAX      = np.array([ 2.4,  4.0,  0.209,  3.5], dtype=np.float32)
ACT_ELEMENTS = np.array([-10.0, 10.0], dtype=np.float32)   # force values for gym actions 0 / 1


# ============================================================
# Q-Network (Critic)
# MATLAB: featureInputLayer(rescale-symmetric) → FC(128)→ReLU→FC(128)→ReLU→FC(2)
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, num_observations: int = 4, num_actions: int = 2):
        super().__init__()
        self.register_buffer('obs_min', torch.tensor(OBS_MIN))
        self.register_buffer('obs_max', torch.tensor(OBS_MAX))

        self.fc1 = nn.Linear(num_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # rescale-symmetric: maps [obs_min, obs_max] → [-1, 1]
        obs_norm = 2.0 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1.0
        x = torch.relu(self.fc1(obs_norm))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# Replay Buffer  (MATLAB: myBuffer struct)
# ============================================================
class ReplayBuffer:
    def __init__(self, buffer_size: int, num_obs: int = 4, num_act: int = 1):
        self.size   = buffer_size
        self.index  = 0        # next write position (0-indexed)
        self.length = 0        # current valid length

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

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.length, size=batch_size)
        return (self.obs[idx], self.act[idx], self.next_obs[idx],
                self.rew[idx], self.done[idx])


# ============================================================
# Custom Reward / Done Function
# MATLAB: rewardIsDoneFunction(nextObs)
# ============================================================
def reward_is_done_function(next_obs):
    """
    next_obs : numpy array (batch, 4)  or  (4,) single step
    Returns  : reward (batch,) float32,  is_done (batch,) bool
    """
    theta_threshold = 12.0 * np.pi / 180.0   # ≈ 0.2094 rad
    x_threshold     = 2.4

    if isinstance(next_obs, torch.Tensor):
        next_obs = next_obs.detach().cpu().numpy()
    next_obs = np.asarray(next_obs, dtype=np.float32)
    if next_obs.ndim == 1:
        next_obs = next_obs.reshape(1, -1)

    x     = next_obs[:, 0]
    theta = next_obs[:, 2]

    is_done    = (np.abs(x) > x_threshold) | (np.abs(theta) > theta_threshold)
    r_angle    = 1.0 - (np.abs(theta) / theta_threshold) ** 2
    r_pos      = 1.0 - (np.abs(x)     / x_threshold)     ** 2
    shaped_rew = 0.4 * 1.0 + 0.4 * r_angle + 0.2 * r_pos

    reward = np.where(is_done, np.float32(-10.0), shaped_rew.astype(np.float32))
    return reward, is_done


# ============================================================
# Main Training Function
# MATLAB: train_DQN_core(runIdx, numEpisodes)
# ============================================================
def train_DQN_core(run_idx: int, num_episodes: int):
    """
    Args
    ----
    run_idx      : random seed (1-based, matching MATLAB)
    num_episodes : total training episodes

    Returns
    -------
    episode_cumulative_reward_vector : list[float]
    episode_step_vector              : list[int]   (cumulative total steps)
    """
    # --------------------------------------------------
    # 1. Environment & parameter setup
    # --------------------------------------------------
    np.random.seed(run_idx)
    torch.manual_seed(run_idx)

    env = gym.make('CartPole-v1')

    num_observations = 4
    num_actions      = 2

    # --------------------------------------------------
    # 2. Critic (Q-Network) initialisation
    # MATLAB: rlVectorQValueFunction / rlOptimizer
    # --------------------------------------------------
    q_network      = QNetwork(num_observations, num_actions)
    target_network = deepcopy(q_network)

    # MATLAB: rlOptimizerOptions('LearnRate',1e-3,'GradientThreshold',1,'L2Reg',0)
    critic_optimizer = optim.Adam(q_network.parameters(), lr=1e-3, weight_decay=0.0)

    # --------------------------------------------------
    # 3. Hyperparameters & buffers
    # MATLAB: myBuffer, maxStepsPerEpisode, discountFactor, etc.
    # --------------------------------------------------
    buffer_size        = int(1e5)
    max_steps_per_ep   = 500
    discount_factor    = 0.99
    mini_batch_size    = 256
    warm_start_samples = 200
    epsilon            = 1.0
    epsilon_min        = 0.01
    epsilon_decay      = 0.005
    num_gradient_steps = 1
    train_interval     = 1      # MATLAB: trainInterval = 1
    tau                = 0.005  # soft-update coefficient

    real_buffer = ReplayBuffer(buffer_size, num_observations, num_act=1)

    episode_cumulative_reward_vector = []
    episode_step_vector              = []
    total_step_ct                    = 0
    best_avg_score                   = -float("inf")

    act_elements_t = torch.tensor(ACT_ELEMENTS)   # [-10., 10.] — used for action mask

    # --------------------------------------------------
    # 4. Training loop
    # --------------------------------------------------
    for episode_ct in range(1, num_episodes + 1):

        obs_raw, _ = env.reset()
        obs        = torch.tensor(obs_raw, dtype=torch.float32).unsqueeze(0)  # (1, 4)
        episode_reward = 0.0

        for step_ct in range(1, max_steps_per_ep + 1):
            total_step_ct += 1

            # 1. Action Selection (epsilon-greedy)
            # MATLAB: if rand() < epsilon → random, else getValue(critic, obs)
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(0, num_actions)
            else:
                with torch.no_grad():
                    q_vals = q_network(obs)
                action_idx = int(q_vals.argmax(dim=1).item())

            action_force = float(ACT_ELEMENTS[action_idx])   # -10 or +10
            gym_action   = action_idx                         # 0 or 1 for gym

            # Epsilon decay (only after warm start)
            # MATLAB: epsilon = max(epsilon * (1 - epsilonDecay), epsilonMin)
            if total_step_ct > warm_start_samples:
                epsilon = max(epsilon * (1.0 - epsilon_decay), epsilon_min)

            # 2. Environment Step
            next_obs_raw, _, terminated, truncated, _ = env.step(gym_action)
            is_done = bool(terminated)

            # Custom shaped reward (MATLAB: rewardIsDoneFunction)
            reward_arr, _ = reward_is_done_function(
                np.array(next_obs_raw, dtype=np.float32).reshape(1, -1)
            )
            reward   = float(reward_arr[0])
            next_obs = torch.tensor(next_obs_raw, dtype=torch.float32).unsqueeze(0)

            # 3. Store Experience
            real_buffer.store(
                obs      = obs.numpy().flatten(),
                act      = np.array([action_force], dtype=np.float32),
                next_obs = next_obs.numpy().flatten(),
                rew      = np.float32(reward),
                done     = is_done,
            )

            episode_reward += reward
            obs = next_obs

            # 4. Train DQN Agent
            # MATLAB: if totalStepCt > warmStartSamples && mod(totalStepCt, trainInterval)==0
            if (total_step_ct > warm_start_samples
                    and (total_step_ct % train_interval) == 0
                    and real_buffer.length >= mini_batch_size):

                for _ in range(num_gradient_steps):
                    s_obs, s_act, s_nxt, s_rew, s_done = real_buffer.sample(mini_batch_size)

                    obs_t      = torch.tensor(s_obs)
                    next_obs_t = torch.tensor(s_nxt)
                    act_t      = torch.tensor(s_act)      # (batch, 1) force values
                    rew_t      = torch.tensor(s_rew)
                    done_t     = torch.tensor(s_done, dtype=torch.bool)

                    # Compute Target Q  (MATLAB: targetCritic / maxNextQ)
                    with torch.no_grad():
                        q_next     = target_network(next_obs_t)
                        max_next_q = q_next.max(dim=1).values   # (batch,)

                    target_q         = rew_t + discount_factor * max_next_q
                    target_q[done_t] = rew_t[done_t]             # terminal: no future

                    # MATLAB: actionMask = (possibleActions == actionBatch)
                    #         qPredicted = sum(qValuesAll .* actionMask, 1)
                    action_mask = (act_elements_t.unsqueeze(0) == act_t).float()  # (batch, 2)
                    q_vals  = q_network(obs_t)
                    q_pred  = (q_vals * action_mask).sum(dim=1)  # (batch,)

                    # MSE loss  (MATLAB: mse(qPredicted, targetQValues, 'DataFormat','CB'))
                    loss = nn.functional.mse_loss(q_pred, target_q)

                    critic_optimizer.zero_grad()
                    loss.backward()
                    # MATLAB: GradientThreshold=1  (value-based clipping)
                    torch.nn.utils.clip_grad_value_(q_network.parameters(), 1.0)
                    critic_optimizer.step()

                    # Soft Update Target Network
                    # MATLAB: tau*w + (1-tau)*t  where w=critic, t=targetCritic
                    with torch.no_grad():
                        for p, tp in zip(q_network.parameters(),
                                         target_network.parameters()):
                            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

            if is_done or truncated:
                break

        episode_cumulative_reward_vector.append(episode_reward)
        episode_step_vector.append(total_step_ct)

        # Champion model save
        if len(episode_cumulative_reward_vector) >= 10:
            current_avg = np.mean(episode_cumulative_reward_vector[-10:])
            if current_avg > best_avg_score and current_avg >= 480:
                best_avg_score = current_avg
                model_path = RESULTS_DIR / f"Champion_Seed{run_idx}_BestModel.pt"
                torch.save(
                    {
                        "q_network": q_network.state_dict(),
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
                f"Eps: {epsilon:.3f}"
            )

    env.close()
    return episode_cumulative_reward_vector, episode_step_vector
