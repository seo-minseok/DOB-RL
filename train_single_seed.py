"""
train_single_seed.py

DOB-MBRL CartPole — 단일 시드 학습 스크립트 (이해용 단순화 버전)
멀티프로세싱, 플롯 코드 제거 / 핵심 로직만 유지
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# Windows + MuJoCo EGL 충돌 방지
if os.name == 'nt' and os.environ.get('MUJOCO_GL', '').lower() == 'egl':
    os.environ.pop('MUJOCO_GL', None)

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym


# ============================================================
# [1] 전역 상수
# ============================================================
X_THRESHOLD    = np.float32(2.4)
THETA_THRESHOLD = np.float32(12.0 * np.pi / 180.0)

OBS_MIN = np.array([-4.8, -5.0, -2 * THETA_THRESHOLD, -5.0], dtype=np.float32)
OBS_MAX = np.array([ 4.8,  5.0,  2 * THETA_THRESHOLD,  5.0], dtype=np.float32)

FORCE_MAG    = np.float32(10.0)
ACT_ELEMENTS = np.array([-FORCE_MAG, FORCE_MAG], dtype=np.float32)  # 이산 행동 → 연속 힘

IN_MIN = np.array([*OBS_MIN, -FORCE_MAG], dtype=np.float32)
IN_MAX = np.array([*OBS_MAX,  FORCE_MAG], dtype=np.float32)

# DOB 선택 행렬
# FPINV: 4차원 상태 오차 → 2차원 (vel, thetadot) 추출
FPINV = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
# F_MAT: 2차원 잔차 → 4차원 상태 갱신
F_MAT = np.array([[0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 1]], dtype=np.float32)


# ============================================================
# [2] 명목 모델 파라미터 & 1스텝 전이
# ============================================================
def default_cartpole_params():
    return {'g': 9.8, 'M': 1.0, 'm': 0.1, 'l': 0.5, 'Ts': 0.02,
            'force_limit': float(FORCE_MAG)}


def step_nominal_cartpole(x, u, p):
    """
    명목 CartPole 1스텝 (단순 Euler):
      - 가속도 항을 0으로 가정 (DOB가 오차를 보정)
      - x_next = x + Ts * [vel, 0, thetadot, 0]
    x: (..., 4), u: (..., 1)
    """
    vel = x[..., 1]
    thd = x[..., 3]
    xdot = np.stack([vel, np.zeros_like(vel), thd, np.zeros_like(thd)], axis=-1)
    return x + p['Ts'] * xdot


# ============================================================
# [3] 신경망 모델
# ============================================================
class QNetwork(nn.Module):
    """
    DQN Q-네트워크: 상태 → 각 행동의 Q값
    입력을 [-1, 1]로 정규화 후 FC(128) → ReLU → FC(128) → ReLU → FC(2)
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('obs_min', torch.tensor(OBS_MIN))
        self.register_buffer('obs_max', torch.tensor(OBS_MAX))
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, obs):
        obs_norm = 2.0 * (obs - self.obs_min) / (self.obs_max - self.obs_min) - 1.0
        return self.net(obs_norm)


class ResidualDxNet(nn.Module):
    """
    잔차 동역학 네트워크: [obs(4), act(1)] → 잔차 변화량(2) [vel_res, thetadot_res]
    DOB의 dhat을 타깃으로 학습
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.register_buffer('in_min', torch.tensor(IN_MIN))
        self.register_buffer('in_max', torch.tensor(IN_MAX))
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        x_norm = 2.0 * (x - self.in_min) / (self.in_max - self.in_min) - 1.0
        return self.net(x_norm)


class NormalizedRBFModel(nn.Module):
    """
    정규화 RBF 불확실성 모델: 현재 상태-행동의 불확실성 크기를 예측
    Centers는 고정(학습 안 함), Weights만 학습
    """
    def __init__(self, num_centers=600, width=0.1, initial_value=5.0):
        super().__init__()
        phys_min = torch.tensor(IN_MIN)
        phys_max = torch.tensor(IN_MAX)

        # 상태 센터: 랜덤 초기화, 행동 센터: -10 / +10 고정
        state_min   = torch.tensor(IN_MIN[:4]).unsqueeze(1)
        state_range = (torch.tensor(IN_MAX[:4]) - torch.tensor(IN_MIN[:4])).unsqueeze(1)
        state_centers = state_min + state_range * torch.rand(4, num_centers)
        half = num_centers // 2
        act_centers = torch.zeros(1, num_centers)
        act_centers[0, :half] = -10.0
        act_centers[0, half:] =  10.0

        raw_centers  = torch.cat([state_centers, act_centers], dim=0)  # (5, K)
        norm_centers = 2.0 * (raw_centers - phys_min.unsqueeze(1)) / (phys_max - phys_min).unsqueeze(1) - 1.0

        self.register_buffer('centers',  norm_centers)
        self.register_buffer('phys_min', phys_min)
        self.register_buffer('phys_max', phys_max)
        self.width   = width
        self.weights = nn.Parameter(torch.ones(2, num_centers) * initial_value)

    def forward(self, x):
        """x: (batch, 5) → (batch, 2) 불확실성 예측"""
        x_norm  = 2.0 * (x - self.phys_min) / (self.phys_max - self.phys_min) - 1.0
        c_sq    = (self.centers ** 2).sum(dim=0)
        x_sq    = (x_norm ** 2).sum(dim=1)
        cross   = x_norm @ self.centers
        dist_sq = c_sq.unsqueeze(0) + x_sq.unsqueeze(1) - 2.0 * cross
        phi     = torch.exp(-dist_sq / (2.0 * self.width ** 2))
        phi_norm = phi / (phi.sum(dim=1, keepdim=True) + 1e-8)
        return phi_norm @ self.weights.t()


# ============================================================
# [4] 리플레이 버퍼 (DOB 전용 필드 포함)
# ============================================================
class ReplayBufferDOB:
    """
    실제/모델 경험을 저장하는 순환 버퍼
    DOB용 추가 필드: dhat (추정 외란), dx_nom (명목 변화), uncertainty (불확실성)
    """
    def __init__(self, buffer_size, num_obs=4, num_act=1):
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
        self.obs[idx], self.act[idx], self.next_obs[idx] = obs, act, next_obs
        self.rew[idx], self.done[idx]                    = rew, done
        self.dhat[idx], self.dx_nom[idx], self.uncertainty[idx] = dhat, dx_nom, uncertainty
        self.index  = (self.index + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def store_batch(self, obs_b, act_b, next_obs_b, rew_b, done_b):
        n       = len(obs_b)
        indices = np.arange(self.index, self.index + n) % self.size
        self.obs[indices], self.act[indices], self.next_obs[indices] = obs_b, act_b, next_obs_b
        self.rew[indices], self.done[indices]                        = rew_b, done_b
        self.index  = (self.index + n) % self.size
        self.length = min(self.length + n, self.size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.length, size=batch_size)
        return (self.obs[idx], self.act[idx], self.next_obs[idx],
                self.rew[idx], self.done[idx])


# ============================================================
# [5] 환경 유틸
# ============================================================
def make_env():
    return gym.make('CartPole-v1')


def reset_env(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, rew, terminated, truncated, info = result
        return obs, rew, bool(terminated or truncated), info
    obs, rew, done, info = result
    return obs, rew, bool(done), info


def reward_is_done(next_obs):
    """커스텀 보상: 각도·위치 기반 shaped reward, 종료 시 -10"""
    if isinstance(next_obs, torch.Tensor):
        next_obs = next_obs.cpu().numpy()
    next_obs = np.asarray(next_obs, dtype=np.float32)
    if next_obs.ndim == 1:
        next_obs = next_obs.reshape(1, -1)
    x, theta = next_obs[:, 0], next_obs[:, 2]
    is_done  = (np.abs(x) > X_THRESHOLD) | (np.abs(theta) > THETA_THRESHOLD)
    r_angle  = 1.0 - (np.abs(theta) / THETA_THRESHOLD) ** 2
    r_pos    = 1.0 - (np.abs(x)     / X_THRESHOLD)     ** 2
    shaped   = 0.4 * 1.0 + 0.4 * r_angle + 0.2 * r_pos
    reward   = np.where(is_done, np.float32(-10.0), shaped.astype(np.float32))
    return reward, is_done


# ============================================================
# [6] 모델 학습 함수
# ============================================================
def train_residual(res_net, optimizer, real_buffer, batch_size, epochs):
    """
    잔차 네트워크 학습: 불확실성 크기로 weighted sampling
    타깃: 버퍼에 저장된 dhat (DOB 추정 외란)
    """
    res_net.train()
    N = real_buffer.length
    weights = np.linalg.norm(real_buffer.uncertainty[:N], axis=1) + 1e-3
    probs   = weights / weights.sum()

    for _ in range(epochs):
        for _ in range(N // batch_size):
            idx    = np.random.choice(N, size=batch_size, replace=True, p=probs)
            inp    = torch.cat([torch.tensor(real_buffer.obs[idx]),
                                torch.tensor(real_buffer.act[idx])], dim=-1)
            target = torch.tensor(real_buffer.dhat[idx])
            loss   = nn.functional.mse_loss(res_net(inp), target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()


def train_uncertainty(uncert_model, optimizer, real_buffer, res_net, batch_size, epochs):
    """
    RBF 불확실성 모델 학습
    타깃: |Fpinv * (실제변화 - 명목변화) - 잔차예측|  (추정 오차 절댓값)
    """
    N = real_buffer.length
    if N == 0:
        return
    obs_all, act_all  = real_buffer.obs[:N], real_buffer.act[:N]
    dx_nom_all        = real_buffer.dx_nom[:N]
    dx_real_all       = real_buffer.next_obs[:N] - obs_all

    with torch.no_grad():
        dx_res_all = res_net(torch.tensor(np.concatenate([obs_all, act_all], axis=-1))).cpu().numpy()

    fresh_uncert = (dx_real_all - dx_nom_all) @ FPINV.T - dx_res_all  # (N, 2)

    uncert_model.train()
    for _ in range(epochs):
        for _ in range(N // batch_size):
            idx    = np.random.randint(0, N, size=batch_size)
            inp    = torch.tensor(np.concatenate([obs_all[idx], act_all[idx]], axis=-1))
            target = torch.tensor(np.abs(fresh_uncert[idx]))
            loss   = nn.functional.mse_loss(uncert_model(inp), target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()


# ============================================================
# [7] 모델 롤아웃 (가상 경험 생성)
# ============================================================
def predict_next_obs(obs, act, res_net, p_nom):
    """명목 1스텝 + 잔차네트워크 보정으로 다음 상태 예측"""
    dx_nom = step_nominal_cartpole(obs, act, p_nom) - obs
    with torch.no_grad():
        dx_res = res_net(torch.tensor(np.concatenate([obs, act], axis=-1))).cpu().numpy()
    return obs + dx_nom + (dx_res @ F_MAT.T)


def generate_rollouts(real_buffer, model_buffer, res_net, uncert_model,
                      q_net, epsilon, opts, p_nom):
    """
    모델 기반 롤아웃으로 가상 경험 생성:
    - 불확실성이 임계값 미만인 경우에만 롤아웃 계속 진행
    - 첫 step(h==0)은 무조건 신뢰 가능 처리
    """
    B         = opts['mini_batch_size']
    horizon   = opts['max_horizon_length']       # 10
    threshold = opts['uncertainty_threshold']    # 0.1
    eps_model = max(epsilon, opts['epsilon_min_model'])

    for _ in range(opts['num_rollout_iter']):    # 20회 반복
        if real_buffer.length < B:
            break
        idx         = np.random.randint(0, real_buffer.length, size=B)
        current_obs = torch.tensor(real_buffer.obs[idx])
        alive       = np.ones(B, dtype=bool)

        for h in range(horizon):
            n_alive = alive.sum()
            if n_alive == 0:
                break

            valid_obs = current_obs[alive].numpy()
            # 행동 선택 (epsilon-greedy)
            act_idx = q_net(torch.tensor(valid_obs)).argmax(dim=1).numpy()
            acts    = ACT_ELEMENTS[act_idx].reshape(-1, 1)
            rand_acts = ACT_ELEMENTS[np.random.randint(0, 2, n_alive)].reshape(-1, 1)
            acts[np.random.rand(n_alive) < eps_model] = rand_acts[np.random.rand(n_alive) < eps_model]

            # 불확실성 체크
            inp_rbf = torch.tensor(np.concatenate([valid_obs, acts], axis=-1))
            with torch.no_grad():
                uncert_mag = np.linalg.norm(uncert_model(inp_rbf).cpu().numpy(), axis=1)
            reliable = uncert_mag < threshold
            if h == 0:
                reliable[:] = True  # 첫 step은 항상 신뢰 가능
            if reliable.sum() == 0:
                break

            rel_obs  = valid_obs[reliable]
            rel_act  = acts[reliable]
            rel_next = predict_next_obs(rel_obs, rel_act, res_net, p_nom)
            rel_rew, rel_done = reward_is_done(rel_next)
            model_buffer.store_batch(rel_obs, rel_act, rel_next, rel_rew, rel_done)

            alive_idx = np.where(alive)[0]
            alive[alive_idx[~reliable]] = False
            rel_idx = alive_idx[reliable]
            alive[rel_idx[rel_done]] = False
            not_done = ~rel_done
            if not_done.any():
                current_obs[rel_idx[not_done]] = torch.tensor(rel_next[not_done])

    return model_buffer


# ============================================================
# [8] 메인 학습 루프
# ============================================================
def train(seed=1, num_episodes=200):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env   = make_env()
    p_nom = default_cartpole_params()

    # --- 네트워크 초기화 ---
    q_net    = QNetwork()
    tgt_net  = deepcopy(q_net)
    res_net  = ResidualDxNet(hidden=32)
    rbf_net  = NormalizedRBFModel(num_centers=600, width=0.1, initial_value=5.0)

    q_opt   = optim.Adam(q_net.parameters(),   lr=1e-3)
    res_opt = optim.SGD(res_net.parameters(),  lr=1e-2, momentum=0.9)
    rbf_opt = optim.SGD(rbf_net.parameters(),  lr=0.5,  momentum=0.9)

    # --- 버퍼 ---
    real_buf  = ReplayBufferDOB(int(1e5))
    model_buf = ReplayBufferDOB(int(1e5))

    # --- 하이퍼파라미터 ---
    batch_size       = 256
    warm_start       = 200    # 이 스텝 이후부터 학습 시작
    num_epochs       = 5
    epsilon          = 1.0
    epsilon_min      = 0.01
    epsilon_decay    = 0.005
    real_ratio       = 0.2    # 미니배치 중 실제 경험 비율
    update_interval  = 10     # N스텝마다 Q-네트워크 업데이트
    num_grad_steps   = 2      # 업데이트당 그래디언트 스텝 수 (UTD)
    tau              = 0.005  # 타깃 네트워크 soft update 계수
    dob_w            = 0.1    # DOB 필터 계수

    rollout_opts = {
        'mini_batch_size'   : batch_size,
        'max_horizon_length': 10,
        'uncertainty_threshold': 0.1,
        'num_rollout_iter'  : 20,
        'epsilon_min_model' : 0.1,
    }

    act_t         = torch.tensor(ACT_ELEMENTS)
    model_trained = False
    total_steps   = 0
    all_rewards   = []

    for ep in range(1, num_episodes + 1):

        # ── Phase 1: 에피소드 시작 전 모델 학습 & 롤아웃 ──────────────
        if real_buf.length > batch_size and total_steps > warm_start:
            train_residual(res_net, res_opt, real_buf, batch_size, num_epochs)
            train_uncertainty(rbf_net, rbf_opt, real_buf, res_net, batch_size, 5)
            model_trained = True
            model_buf = generate_rollouts(real_buf, model_buf, res_net, rbf_net,
                                          q_net, epsilon, rollout_opts, p_nom)

        # ── Phase 2: 에피소드 초기화 ──────────────────────────────────
        obs        = np.array(reset_env(env), dtype=np.float32)
        ep_reward  = 0.0
        dhat       = np.zeros(2, dtype=np.float32)  # DOB 추정 외란 (에피소드마다 리셋)

        # ── Phase 3: 환경과 상호작용 ──────────────────────────────────
        for step in range(1, 501):
            total_steps += 1

            # Epsilon-greedy 행동 선택
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(0, 2)
            else:
                with torch.no_grad():
                    a_idx = int(q_net(torch.tensor(obs).unsqueeze(0)).argmax().item())

            force = float(ACT_ELEMENTS[a_idx])

            if total_steps > warm_start:
                epsilon = max(epsilon * (1.0 - epsilon_decay), epsilon_min)

            next_obs_raw, _, is_done, _ = step_env(env, a_idx)
            next_obs = np.array(next_obs_raw, dtype=np.float32)

            # ── DOB 온라인 업데이트 ────────────────────────────────────
            # 실제 상태 변화
            dx_real = next_obs - obs

            # 명목 상태 변화
            dx_nom  = step_nominal_cartpole(obs.reshape(1, -1),
                                            np.array([[force]], dtype=np.float32),
                                            p_nom).flatten() - obs

            # 잔차 네트워크 예측
            with torch.no_grad():
                inp_res = torch.tensor(np.concatenate([obs, [force]], dtype=np.float32)).unsqueeze(0)
                dx_res  = res_net(inp_res).cpu().numpy().flatten()

            # 오차 계산 및 DOB 추정
            e    = dx_real - dx_nom
            dhat = dob_w * dx_res + (1.0 - dob_w) * (FPINV @ e)  # DOB 필터

            # 불확실성: 현재 모델이 설명하지 못하는 부분
            uncertainty = FPINV @ e - dx_res

            reward_arr, _ = reward_is_done(next_obs.reshape(1, -1))
            reward = float(reward_arr[0])

            real_buf.store(obs, np.array([force], dtype=np.float32),
                           next_obs, np.float32(reward), is_done,
                           dhat, dx_nom, uncertainty)

            ep_reward += reward
            obs = next_obs

            # ── Phase 4: Q-네트워크 업데이트 (UTD) ────────────────────
            if (step % update_interval == 0) and (total_steps > warm_start):
                for _ in range(num_grad_steps * update_interval):
                    if real_buf.length < batch_size:
                        break

                    # 실제 + 모델 경험 혼합 샘플링
                    if model_trained and real_ratio < 1.0:
                        n_real  = int(np.ceil(real_ratio * batch_size))
                        n_model = batch_size - n_real
                        s_o, s_a, s_n, s_r, s_d = [
                            np.concatenate([x, y], axis=0)
                            for x, y in zip(real_buf.sample(n_real),
                                            model_buf.sample(n_model))
                        ]
                    else:
                        s_o, s_a, s_n, s_r, s_d = real_buf.sample(batch_size)

                    obs_t  = torch.tensor(s_o)
                    nxt_t  = torch.tensor(s_n)
                    act_t_ = torch.tensor(s_a)
                    rew_t  = torch.tensor(s_r)
                    don_t  = torch.tensor(s_d, dtype=torch.bool)

                    with torch.no_grad():
                        max_q_next = tgt_net(nxt_t).max(dim=1).values

                    # Bellman 타깃
                    target_q          = rew_t + 0.99 * max_q_next
                    target_q[don_t]   = rew_t[don_t]

                    # 행동에 해당하는 Q값만 선택 (act_t: ACT_ELEMENTS 텐서, act_t_: 배치 행동값)
                    act_mask = (act_t.unsqueeze(0) == act_t_).float()
                    q_pred   = (q_net(obs_t) * act_mask).sum(dim=1)

                    loss = nn.functional.mse_loss(q_pred, target_q)
                    q_opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_value_(q_net.parameters(), 1.0)
                    q_opt.step()

                    # Soft target update
                    with torch.no_grad():
                        for p, tp in zip(q_net.parameters(), tgt_net.parameters()):
                            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

            if is_done:
                break

        all_rewards.append(ep_reward)
        avg10 = np.mean(all_rewards[-10:]) if len(all_rewards) >= 10 else ep_reward
        print(f'[Seed {seed}] Ep {ep:3d} | Reward: {ep_reward:6.1f} | '
              f'Avg10: {avg10:6.1f} | ε: {epsilon:.3f} | Steps: {total_steps}')

    env.close()
    return all_rewards


# ============================================================
# [9] 실행 진입점
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DOB-MBRL CartPole 단일 시드 학습')
    parser.add_argument('--seed',     type=int, default=1,   help='랜덤 시드')
    parser.add_argument('--episodes', type=int, default=200, help='총 에피소드 수')
    args = parser.parse_args()

    rewards = train(seed=args.seed, num_episodes=args.episodes)
    print(f'\n학습 완료 | 마지막 10 에피소드 평균: {np.mean(rewards[-10:]):.1f}')
