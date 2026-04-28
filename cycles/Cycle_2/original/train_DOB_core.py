"""
train_DOB_core.py

PyTorch + Gymnasium implementation of train_DOB_core.m
All network architectures, hyperparameters, and DOB logic are identical to the MATLAB version.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from copy import deepcopy

if os.name == 'nt' and os.environ.get('MUJOCO_GL', '').lower() == 'egl':
    os.environ.pop('MUJOCO_GL', None)

try:
    import gymnasium as gym
except ModuleNotFoundError:
    try:
        import gym
    except ModuleNotFoundError:
        gym = None

# ============================================================
# Constants aligned to Gymnasium CartPole-v1
# Note: cart velocity and pole angular velocity are unbounded in the
# environment observation space, so we keep finite proxy limits for
# network normalization.
# ============================================================
X_THRESHOLD = np.float32(2.4)
THETA_THRESHOLD = np.float32(12.0 * np.pi / 180.0)
OBS_CART_POSITION_LIMIT = np.float32(2.0 * X_THRESHOLD)
OBS_POLE_ANGLE_LIMIT = np.float32(2.0 * THETA_THRESHOLD)
OBS_CART_VELOCITY_LIMIT = np.float32(5.0)
OBS_POLE_ANGULAR_VELOCITY_LIMIT = np.float32(5.0)
FORCE_MAG = np.float32(10.0)

OBS_MIN = np.array([
    -OBS_CART_POSITION_LIMIT,
    -OBS_CART_VELOCITY_LIMIT,
    -OBS_POLE_ANGLE_LIMIT,
    -OBS_POLE_ANGULAR_VELOCITY_LIMIT,
], dtype=np.float32)
OBS_MAX = np.array([
    OBS_CART_POSITION_LIMIT,
    OBS_CART_VELOCITY_LIMIT,
    OBS_POLE_ANGLE_LIMIT,
    OBS_POLE_ANGULAR_VELOCITY_LIMIT,
], dtype=np.float32)
ACT_ELEMENTS = np.array([-FORCE_MAG, FORCE_MAG], dtype=np.float32)

# MATLAB: Fpinv = [0 1 0 0; 0 0 0 1]  — extracts velocity & theta_dot change
FPINV = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)

# MATLAB: F = [0 0; 1 0; 0 0; 0 1]  — maps 2D residual → 4D state update
F_MAT = np.array([[0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 1]], dtype=np.float32)


# ============================================================
# CartPole Nominal Parameters  (MATLAB: defaultCartPoleParams)
# ============================================================
def default_cartpole_params():
    return {
        'g': 9.8,
        'M': 1.0,
        'm': 0.1,
        'l': 0.5,
        'Ts': 0.02,
        'force_limit': float(FORCE_MAG),
    }


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
        obs_norm = 2.0 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1.0
        x = torch.relu(self.fc1(obs_norm))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# Residual Dynamics Network
# MATLAB: initResidualDxNet(nObs=4, nAct=1, hidden=32)
# Input: [obs, act] = 5-dim,  Output: 2-dim (vel & theta_dot residuals)
# ============================================================
IN_MIN = np.array([*OBS_MIN, -FORCE_MAG], dtype=np.float32)
IN_MAX = np.array([*OBS_MAX, FORCE_MAG], dtype=np.float32)

class ResidualDxNet(nn.Module):
    def __init__(self, num_obs: int = 4, num_act: int = 1, hidden: int = 32):
        super().__init__()
        self.register_buffer('in_min', torch.tensor(IN_MIN))
        self.register_buffer('in_max', torch.tensor(IN_MAX))
        self.fc1 = nn.Linear(num_obs + num_act, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)   # MATLAB: fullyConnectedLayer(2,"Name","dx_res")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 5)
        x_norm = 2.0 * (x - self.in_min) / (self.in_max - self.in_min) - 1.0
        x = torch.relu(self.fc1(x_norm))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)   # (batch, 2)


# ============================================================
# Normalized RBF Uncertainty Model
# MATLAB: initNormalizedRBF_Structured + forwardNormalizedRBF
# Only model.Weights are trainable; Centers are fixed at init.
# ============================================================
PHYS_MIN = np.array([*OBS_MIN, -FORCE_MAG], dtype=np.float32)
PHYS_MAX = np.array([*OBS_MAX, FORCE_MAG], dtype=np.float32)

class NormalizedRBFModel(nn.Module):
    def __init__(self, num_centers: int = 600, width: float = 0.1,
                 initial_value: float = 5.0):
        super().__init__()
        phys_min_t = torch.tensor(PHYS_MIN)
        phys_max_t = torch.tensor(PHYS_MAX)

        # MATLAB: stateCenters = physMin(1:4) + (physMax(1:4)-physMin(1:4)).*rand(4,K)
        state_min = torch.tensor(PHYS_MIN[:4]).unsqueeze(1)
        state_range = (torch.tensor(PHYS_MAX[:4]) - torch.tensor(PHYS_MIN[:4])).unsqueeze(1)
        state_centers = state_min + state_range * torch.rand(4, num_centers)
        half = num_centers // 2
        act_centers = torch.zeros(1, num_centers)
        act_centers[0, :half]  = -10.0   # MATLAB: actCenters(1,1:half) = -10
        act_centers[0, half:]  =  10.0   # MATLAB: actCenters(1,half+1:end) = 10

        raw_centers  = torch.cat([state_centers, act_centers], dim=0)  # (5, K)
        norm_centers = (2.0 * (raw_centers - phys_min_t.unsqueeze(1)) /
                        (phys_max_t - phys_min_t).unsqueeze(1) - 1.0)

        self.register_buffer('centers',  norm_centers)   # (5, K) — not trained
        self.register_buffer('phys_min', phys_min_t)
        self.register_buffer('phys_max', phys_max_t)
        self.width   = width
        self.weights = nn.Parameter(torch.ones(2, num_centers) * initial_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 5)  — raw [obs, act]
        Returns (batch, 2)  — predicted uncertainty
        """
        x_norm  = (2.0 * (x - self.phys_min) / (self.phys_max - self.phys_min) - 1.0)
        # MATLAB: distSq=(sum(C.^2,1)'+sum(xN.^2,1))-2*(C'*xN)  → (K,batch)
        c_sq    = (self.centers ** 2).sum(dim=0)          # (K,)
        x_sq    = (x_norm ** 2).sum(dim=1)                # (batch,)
        cross   = x_norm @ self.centers                    # (batch, K)
        dist_sq = c_sq.unsqueeze(0) + x_sq.unsqueeze(1) - 2.0 * cross  # (batch, K)
        phi     = torch.exp(-dist_sq / (2.0 * self.width ** 2))
        phi_norm = phi / (phi.sum(dim=1, keepdim=True) + 1e-8)
        return phi_norm @ self.weights.t()                 # (batch, 2)


# ============================================================
# Replay Buffer (DOB-extended)
# MATLAB: myBuffer struct with dhat, dxNom, uncertainty fields
# ============================================================
class ReplayBufferDOB:
    def __init__(self, buffer_size: int, num_obs: int = 4, num_act: int = 1):
        self.size   = buffer_size
        self.index  = 0
        self.length = 0

        self.obs         = np.zeros((buffer_size, num_obs), dtype=np.float32)
        self.next_obs    = np.zeros((buffer_size, num_obs), dtype=np.float32)
        self.act         = np.zeros((buffer_size, num_act), dtype=np.float32)
        self.rew         = np.zeros(buffer_size,            dtype=np.float32)
        self.done        = np.zeros(buffer_size,            dtype=bool)
        self.dhat        = np.zeros((buffer_size, 2),       dtype=np.float32)
        self.dx_nom      = np.zeros((buffer_size, num_obs), dtype=np.float32)
        self.uncertainty = np.zeros((buffer_size, 2),       dtype=np.float32)

    def store(self, obs, act, next_obs, rew, done, dhat, dx_nom, uncertainty):
        idx = self.index
        self.obs[idx]         = obs
        self.act[idx]         = act
        self.next_obs[idx]    = next_obs
        self.rew[idx]         = rew
        self.done[idx]        = done
        self.dhat[idx]        = dhat
        self.dx_nom[idx]      = dx_nom
        self.uncertainty[idx] = uncertainty
        self.index  = (self.index + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def store_batch(self, obs_b, act_b, next_obs_b, rew_b, done_b):
        """Batch store with wrap-around (model buffer use)."""
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


# ============================================================
# Nominal CartPole Dynamics  (MATLAB: stepNominalCartPole)
# Simplified Euler: xNomNext = x + Ts * [vel, 0, thd, 0]
# Accelerations (posdd, thdd) are set to zero in the nominal model —
# the DOB estimates and compensates for the real dynamics.
# ============================================================
def step_nominal_cartpole(x: np.ndarray, u: np.ndarray, p: dict) -> np.ndarray:
    """
    x : (..., 4)  numpy  — [pos, vel, theta, thetadot]
    u : (..., 1)  numpy  — force
    Returns xNomNext with same shape as x.
    """
    vel  = x[..., 1]
    thd  = x[..., 3]
    xdot = np.stack([vel,
                     np.zeros_like(vel),
                     thd,
                     np.zeros_like(thd)], axis=-1)
    return x + p['Ts'] * xdot


# ============================================================
# Reward / Done Function  (MATLAB: rewardIsDoneFunction)
# ============================================================
def reward_is_done_function(next_obs):
    if isinstance(next_obs, torch.Tensor):
        next_obs = next_obs.detach().cpu().numpy()
    next_obs = np.asarray(next_obs, dtype=np.float32)
    if next_obs.ndim == 1:
        next_obs = next_obs.reshape(1, -1)

    x     = next_obs[:, 0]
    theta = next_obs[:, 2]

    is_done    = (np.abs(x) > X_THRESHOLD) | (np.abs(theta) > THETA_THRESHOLD)
    r_angle    = 1.0 - (np.abs(theta) / THETA_THRESHOLD) ** 2
    r_pos      = 1.0 - (np.abs(x)     / X_THRESHOLD)     ** 2
    shaped_rew = 0.4 * 1.0 + 0.4 * r_angle + 0.2 * r_pos
    reward     = np.where(is_done, np.float32(-10.0), shaped_rew.astype(np.float32))
    return reward, is_done


def make_cartpole_env():
    if gym is None:
        raise ModuleNotFoundError(
            "Neither 'gymnasium' nor 'gym' is installed. "
            "Install one of them before running train_DOB_core, for example: "
            "python -m pip install gymnasium[classic-control]"
        )
    return gym.make('CartPole-v1')


def reset_env(env):
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def step_env(env, action):
    step_result = env.step(action)
    if len(step_result) == 5:
        next_obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return next_obs, reward, done, info

    if len(step_result) == 4:
        next_obs, reward, done, info = step_result
        return next_obs, reward, bool(done), info

    raise RuntimeError(
        f"Unexpected env.step return length: {len(step_result)}"
    )


# ============================================================
# Predict Next Observation (DOB)
# MATLAB: predictNextObsDOB
# nextObs = obs + dxNom + F * dxRes
# ============================================================
def predict_next_obs_dob(obs: np.ndarray, act: np.ndarray,
                          res_net: ResidualDxNet, p_nom: dict,
                          use_nominal: bool) -> np.ndarray:
    """
    obs  : (batch, 4) numpy
    act  : (batch, 1) numpy  — continuous force
    Returns nextObs (batch, 4) numpy
    """
    if use_nominal:
        x_nom_next = step_nominal_cartpole(obs, act, p_nom)
        dx_nom     = x_nom_next - obs          # (batch, 4)
    else:
        dx_nom = np.zeros_like(obs)

    with torch.no_grad():
        inp    = torch.tensor(np.concatenate([obs, act], axis=-1))
        dx_res = res_net(inp).cpu().numpy()    # (batch, 2)

    next_obs = obs + dx_nom + (dx_res @ F_MAT.T)   # (batch, 4)
    return next_obs


# ============================================================
# Train Residual Dynamics Model (DOB)
# MATLAB: trainResidualDxModelDOB
# Uses uncertainty-weighted sampling (randsample with weights)
# Target: model.dhat  (DOB disturbance estimate stored in buffer)
# ============================================================
def train_residual_dx_model_dob(res_net: ResidualDxNet,
                                 optimizer: optim.Optimizer,
                                 real_buffer: ReplayBufferDOB,
                                 mini_batch_size: int,
                                 num_epochs: int) -> float:
    res_net.train()
    valid_len = real_buffer.length

    # MATLAB: uncertMag = sqrt(sum(uncertainty.^2,1))
    uncert_mag = np.linalg.norm(real_buffer.uncertainty[:valid_len], axis=1)  # (N,)
    # MATLAB: samplingWeights = uncertMag + 1e-3
    weights = uncert_mag + 1e-3
    probs   = weights / weights.sum()

    loss_sum = 0.0
    loss_ct  = 0

    for _ in range(num_epochs):
        num_iterations = valid_len // mini_batch_size
        for _ in range(num_iterations):
            # MATLAB: randsample(validLen, miniBatchSize, true, samplingWeights)
            idx     = np.random.choice(valid_len, size=mini_batch_size,
                                       replace=True, p=probs)
            obs_t   = torch.tensor(real_buffer.obs[idx])
            act_t   = torch.tensor(real_buffer.act[idx])
            dhat_t  = torch.tensor(real_buffer.dhat[idx])   # (batch, 2) target

            inp    = torch.cat([obs_t, act_t], dim=-1)
            dx_res = res_net(inp)                            # (batch, 2)
            loss   = nn.functional.mse_loss(dx_res, dhat_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            loss_ct  += 1

    return loss_sum / max(1, loss_ct)


# ============================================================
# Train Uncertainty RBF Model
# MATLAB: trainUncertaintyRBF
# Recomputes fresh target = |Fpinv*e - dxRes| once before training loop.
# ============================================================
def train_uncertainty_rbf(uncert_model: NormalizedRBFModel,
                           optimizer: optim.Optimizer,
                           real_buffer: ReplayBufferDOB,
                           res_net: ResidualDxNet,
                           batch_size: int,
                           epochs: int) -> float:
    valid_len = real_buffer.length
    if valid_len == 0:
        return float('nan')

    # MATLAB: Compute fresh_uncertainty_all once before the training loop
    obs_all      = real_buffer.obs[:valid_len]
    act_all      = real_buffer.act[:valid_len]
    next_obs_all = real_buffer.next_obs[:valid_len]
    dx_nom_all   = real_buffer.dx_nom[:valid_len]

    with torch.no_grad():
        dl_in_all  = torch.tensor(np.concatenate([obs_all, act_all], axis=-1))
        dx_res_all = res_net(dl_in_all).cpu().numpy()    # (N, 2)

    dx_real_all    = next_obs_all - obs_all              # (N, 4)
    e_all          = dx_real_all - dx_nom_all            # (N, 4)
    fpinv_e        = e_all @ FPINV.T                     # (N, 2)
    # MATLAB: fresh_uncertainty_all = Fpinv * eAll - dxResAll_current
    fresh_uncert   = fpinv_e - dx_res_all                # (N, 2)

    loss_sum = 0.0
    ct       = 0

    uncert_model.train()
    for _ in range(epochs):
        num_iter = valid_len // batch_size
        if num_iter == 0:
            break
        for _ in range(num_iter):
            idx      = np.random.randint(0, valid_len, size=batch_size)
            inp_t    = torch.tensor(
                np.concatenate([obs_all[idx], act_all[idx]], axis=-1))
            # MATLAB: target = abs(dlarray(fresh_uncertainty_all(:,idx),"CB"))
            target_t = torch.tensor(np.abs(fresh_uncert[idx]))

            pred = uncert_model(inp_t)                   # (batch, 2)
            loss = nn.functional.mse_loss(pred, target_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            ct       += 1

    return loss_sum / max(1, ct)


# ============================================================
# Generate Rollout Samples (DOB)
# MATLAB: generateSamplesDOB
# Rollout with per-step uncertainty gating; h==0 → always reliable.
# ============================================================
def generate_samples_dob(real_buffer: ReplayBufferDOB,
                          model_buffer: ReplayBufferDOB,
                          res_net: ResidualDxNet,
                          uncert_model: NormalizedRBFModel,
                          q_network: QNetwork,
                          epsilon: float,
                          options: dict,
                          p_nom: dict,
                          use_nominal: bool) -> ReplayBufferDOB:
    max_horizon      = options['max_horizon_length']            # 10
    uncert_threshold = options['uncertainty_threshold']         # 0.1
    num_iter         = options['num_generate_sample_iteration'] # 20
    B                = options['mini_batch_size']
    # MATLAB: epsModel = max(epsilon, epsilonMinModel)
    eps_model        = max(epsilon, options['epsilon_min_model'])

    for _ in range(num_iter):
        if real_buffer.length < B:
            return model_buffer

        # MATLAB: randi(currentBufferLength, [1,B])  — uniform (not weighted)
        idx         = np.random.randint(0, real_buffer.length, size=B)
        current_obs = torch.tensor(real_buffer.obs[idx])   # (B, 4)
        alive_mask  = np.ones(B, dtype=bool)

        for h in range(max_horizon):
            n_alive = alive_mask.sum()
            if n_alive == 0:
                break

            valid_obs = current_obs[alive_mask]   # (n_alive, 4)

            # Greedy action from Q-network
            with torch.no_grad():
                action_idx = q_network(valid_obs).argmax(dim=1).numpy()
            valid_act = ACT_ELEMENTS[action_idx].reshape(-1, 1)  # (n_alive, 1)

            # Epsilon-greedy replacement
            rand_act      = ACT_ELEMENTS[np.random.randint(0, 2, n_alive)].reshape(-1, 1)
            replace_mask  = np.random.rand(n_alive) < eps_model
            valid_act[replace_mask] = rand_act[replace_mask]

            # Uncertainty check
            valid_obs_np = valid_obs.cpu().numpy()
            inp_rbf = torch.tensor(
                np.concatenate([valid_obs_np, valid_act], axis=-1))
            with torch.no_grad():
                pred_uncert  = uncert_model(inp_rbf).cpu().numpy()  # (n_alive, 2)
            uncert_mag_arr   = np.linalg.norm(pred_uncert, axis=1)  # (n_alive,)
            is_reliable      = uncert_mag_arr < uncert_threshold

            # MATLAB: if h <= 1, isReliable(:) = true  (1-indexed, so h==0 here)
            if h == 0:
                is_reliable[:] = True

            n_reliable = is_reliable.sum()
            if n_reliable == 0:
                break

            rel_obs    = valid_obs_np[is_reliable]
            rel_act    = valid_act[is_reliable]
            rel_next   = predict_next_obs_dob(rel_obs, rel_act, res_net, p_nom, use_nominal)
            rel_rew, rel_done = reward_is_done_function(rel_next)

            model_buffer.store_batch(rel_obs, rel_act, rel_next, rel_rew, rel_done)

            # Update alive_mask
            alive_idx = np.where(alive_mask)[0]

            # Kill unreliable
            alive_mask[alive_idx[~is_reliable]] = False

            # Kill reliable-but-done
            reliable_idx = alive_idx[is_reliable]
            alive_mask[reliable_idx[rel_done]] = False

            # Advance observations for survivors
            not_done = ~rel_done
            if not_done.any():
                to_update = reliable_idx[not_done]
                current_obs[to_update] = torch.tensor(rel_next[not_done])

    return model_buffer


# ============================================================
# Sample Mixed Minibatch  (MATLAB: sampleMinibatch)
# realRatio=0.2  →  20% real, 80% model
# ============================================================
def sample_mixed_minibatch(model_trained: bool, real_ratio: float,
                            mini_batch_size: int,
                            real_buffer: ReplayBufferDOB,
                            model_buffer: ReplayBufferDOB):
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


# ============================================================
# Main Training Function
# MATLAB: train_DOB_core(runIdx, numEpisodes, dataQueue)
# ============================================================
def train_DOB_core(run_idx: int, num_episodes: int, result_queue=None):
    """
    Args
    ----
    run_idx      : random seed (1-based, matching MATLAB runIdx)
    num_episodes : total training episodes
    result_queue : multiprocessing.Queue for live progress (MATLAB: dataQueue)

    Returns
    -------
    episode_cumulative_reward_vector : list[float]
    episode_step_vector              : list[int]  (cumulative total steps)
    """
    # --------------------------------------------------
    # 1. Initialisation
    # --------------------------------------------------
    np.random.seed(run_idx)
    torch.manual_seed(run_idx)

    env = make_cartpole_env()

    num_observations = 4
    num_actions      = 2
    num_act_features = 1   # MATLAB: numContinuousActions = 1

    p_nom       = default_cartpole_params()
    use_nominal = True
    use_dob     = True
    dob_w       = 0.1                            # MATLAB: DOB.w
    dhat        = np.zeros(2, dtype=np.float32)  # MATLAB: DOB.dhat

    # --------------------------------------------------
    # 2. Q-Network (Critic)
    # --------------------------------------------------
    q_network      = QNetwork(num_observations, num_actions)
    target_network = deepcopy(q_network)
    # MATLAB: rlOptimizerOptions('LearnRate',1e-3,'GradientThreshold',1)
    critic_opt = optim.Adam(q_network.parameters(), lr=1e-3, weight_decay=0.0)

    # --------------------------------------------------
    # 3. Residual & Uncertainty Models
    # --------------------------------------------------
    res_net     = ResidualDxNet(num_observations, num_act_features, hidden=32)
    # MATLAB: sgdmupdate(lr=1e-2, momentum=0.9)
    res_net_opt = optim.SGD(res_net.parameters(), lr=1e-2, momentum=0.9)

    uncert_model = NormalizedRBFModel(num_centers=600, width=0.1, initial_value=5.0)
    # MATLAB: uncertLearnRate=0.5, momentum=0.9
    rbf_opt = optim.SGD(uncert_model.parameters(), lr=0.5, momentum=0.9)

    # --------------------------------------------------
    # 4. Buffers
    # --------------------------------------------------
    buffer_size  = int(1e5)
    real_buffer  = ReplayBufferDOB(buffer_size, num_observations, num_act_features)
    model_buffer = ReplayBufferDOB(buffer_size, num_observations, num_act_features)

    # --------------------------------------------------
    # 5. Hyperparameters
    # --------------------------------------------------
    max_steps_per_ep   = 500
    discount_factor    = 0.99
    mini_batch_size    = 256
    warm_start_samples = 200
    num_epochs         = 5
    epsilon            = 1.0
    epsilon_min        = 0.01
    epsilon_decay      = 0.005
    real_ratio         = 0.2
    num_gradient_steps = 2
    update_interval    = 10
    tau                = 0.005    # soft-update coefficient

    sample_gen_options = {
        'max_horizon_length'            : 10,   # MATLAB: maxHorizonLength=10
        'uncertainty_threshold'         : 0.1,  # MATLAB: uncertaintyThreshold=0.1
        'num_generate_sample_iteration' : 20,
        'mini_batch_size'               : mini_batch_size,
        'num_observations'              : num_observations,
        'epsilon_min_model'             : 0.1,
    }

    episode_cumulative_reward_vector = []
    episode_step_vector              = []
    total_step_ct                    = 0
    model_trained_at_least_once      = False
    best_avg_score                   = -float('inf')

    act_elements_t = torch.tensor(ACT_ELEMENTS)

    # --------------------------------------------------
    # 6. Main Training Loop
    # --------------------------------------------------
    for episode_ct in range(1, num_episodes + 1):

        # [Phase 1] Model Training & Rollout
        if real_buffer.length > mini_batch_size and total_step_ct > warm_start_samples:
            if real_ratio < 1.0:
                train_residual_dx_model_dob(
                    res_net, res_net_opt, real_buffer,
                    mini_batch_size, num_epochs
                )
                train_uncertainty_rbf(
                    uncert_model, rbf_opt, real_buffer, res_net,
                    mini_batch_size, 5
                )
                model_trained_at_least_once = True
                model_buffer = generate_samples_dob(
                    real_buffer, model_buffer, res_net, uncert_model,
                    q_network, epsilon, sample_gen_options, p_nom, use_nominal
                )

        # [Phase 2] Episode Reset
        obs = np.array(reset_env(env), dtype=np.float32)
        obs_t      = torch.tensor(obs).unsqueeze(0)      # (1, 4)
        episode_reward = 0.0
        dhat           = np.zeros(2, dtype=np.float32)   # reset per episode

        # [Phase 3] Environment Interaction
        for step_ct in range(1, max_steps_per_ep + 1):
            total_step_ct += 1

            # Epsilon-greedy action
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(0, num_actions)
            else:
                with torch.no_grad():
                    action_idx = int(q_network(obs_t).argmax(dim=1).item())

            action_force = float(ACT_ELEMENTS[action_idx])
            gym_action   = action_idx

            if total_step_ct > warm_start_samples:
                epsilon = max(epsilon * (1.0 - epsilon_decay), epsilon_min)

            next_obs_raw, _, is_done, _ = step_env(env, gym_action)
            next_obs = np.array(next_obs_raw, dtype=np.float32)

            # DOB online update
            dx_real = next_obs - obs                          # (4,)
            if use_nominal:
                x_nom_next = step_nominal_cartpole(
                    obs.reshape(1, -1),
                    np.array([[action_force]], dtype=np.float32),
                    p_nom
                ).flatten()
                dx_nom = x_nom_next - obs                     # (4,)
            else:
                dx_nom = np.zeros_like(obs)

            with torch.no_grad():
                inp_res = torch.tensor(
                    np.concatenate([obs, [action_force]], dtype=np.float32)
                ).unsqueeze(0)
                dx_res = res_net(inp_res).cpu().numpy().flatten()   # (2,)

            e = dx_real - dx_nom                              # (4,)
            if use_dob:
                # MATLAB: DOB.dhat = DOB.w*dxRes + (1-DOB.w)*Fpinv*e
                dhat = dob_w * dx_res + (1.0 - dob_w) * (FPINV @ e)
            else:
                dhat = np.zeros(2, dtype=np.float32)

            uncertainty = FPINV @ e - dx_res                  # (2,)

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
            # MATLAB: mod(stepCt,updateInterval)==0, totalUpdatesToRun=numGradientSteps*updateInterval
            if (step_ct % update_interval == 0) and (total_step_ct > warm_start_samples):
                total_updates = num_gradient_steps * update_interval

                for _ in range(total_updates):
                    if real_buffer.length < mini_batch_size:
                        break
                    s_obs, s_act, s_nxt, s_rew, s_done = sample_mixed_minibatch(
                        model_trained_at_least_once, real_ratio,
                        mini_batch_size, real_buffer, model_buffer
                    )

                    obs_bt  = torch.tensor(s_obs)
                    nxt_bt  = torch.tensor(s_nxt)
                    act_bt  = torch.tensor(s_act)   # (batch, 1) force values
                    rew_bt  = torch.tensor(s_rew)
                    done_bt = torch.tensor(s_done, dtype=torch.bool)

                    with torch.no_grad():
                        max_next_q = target_network(nxt_bt).max(dim=1).values

                    target_q          = rew_bt + discount_factor * max_next_q
                    target_q[done_bt] = rew_bt[done_bt]

                    # MATLAB: actionIndicationMatrix = (possibleActions == actionBatch)
                    act_mask = (act_elements_t.unsqueeze(0) == act_bt).float()
                    q_pred   = (q_network(obs_bt) * act_mask).sum(dim=1)
                    loss     = nn.functional.mse_loss(q_pred, target_q)

                    critic_opt.zero_grad()
                    loss.backward()
                    # MATLAB: GradientThreshold=1
                    torch.nn.utils.clip_grad_value_(q_network.parameters(), 1.0)
                    critic_opt.step()

                    # Soft target update  (MATLAB: tau*w + (1-tau)*t)
                    with torch.no_grad():
                        for p, tp in zip(q_network.parameters(),
                                         target_network.parameters()):
                            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

            if is_done:
                break

        # Record episode result
        episode_cumulative_reward_vector.append(episode_reward)
        episode_step_vector.append(total_step_ct)

        # Live update  (MATLAB: send(dataQueue, struct(...)))
        if result_queue is not None:
            result_queue.put({
                'run_idx': run_idx,
                'ep_idx' : episode_ct,
                'reward' : episode_reward,
                'step'   : total_step_ct,
            })

        # Save best model  (MATLAB: Champion_Seed%d_BestModel.mat)
        if len(episode_cumulative_reward_vector) >= 10:
            current_avg = np.mean(episode_cumulative_reward_vector[-10:])
            if current_avg > best_avg_score and current_avg >= 480:
                best_avg_score = current_avg
                torch.save({
                    'q_network'    : q_network.state_dict(),
                    'res_net'      : res_net.state_dict(),
                    'uncert_model' : uncert_model.state_dict(),
                    'total_steps'  : total_step_ct,
                    'episode'      : episode_ct,
                }, f'Champion_Seed{run_idx}_BestModel.pt')
                print(f'[Seed {run_idx}] New best! Avg {current_avg:.1f} '
                      f'at ep {episode_ct} / step {total_step_ct}')

    env.close()
    return episode_cumulative_reward_vector, episode_step_vector