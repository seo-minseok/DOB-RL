"""
test_DOB_core.py — 저장된 모델을 로드해 greedy 평가 실행
trainer.py의 학습 루프와 동일한 DOB 계산을 유지하되,
탐색(epsilon) 없이 스텝 단위 데이터를 전부 기록한다.

저장 형식: 스텝 단위 CSV 1개 (에피소드 요약은 groupby로 유도 가능)
"""
import os
import numpy as np
import torch
from copy import deepcopy

from dob_mbrl.training.config import DOBMBRLConfig
from dob_mbrl.models import QNetwork, ResidualDxNet, NormalizedRBFModel
from dob_mbrl.dynamics import (
    default_cartpole_params, step_nominal_cartpole,
    ACT_ELEMENTS, FPINV, F_MAT,
)
from dob_mbrl.envs.cartpole_utils import (
    make_cartpole_env, reset_env, step_env, reward_is_done_function,
)


def test_DOB_core(
    run_idx: int,
    checkpoint_dir: str,
    num_test_episodes: int = 30,
    cfg: DOBMBRLConfig = None,
) -> list[dict]:
    """
    저장된 체크포인트를 로드해 greedy 정책으로 평가한다.

    Parameters
    ----------
    run_idx           : 시드 번호 (체크포인트 파일명에 사용)
    checkpoint_dir    : Champion_Seed{run_idx}_BestModel.pt 가 있는 디렉토리
    num_test_episodes : 평가 에피소드 수
    cfg               : DOBMBRLConfig (None이면 기본값)

    Returns
    -------
    step_logs : list[dict]
        스텝마다 기록된 딕셔너리 리스트.
        컬럼: seed, episode, step,
              obs_x, obs_xdot, obs_theta, obs_thetadot,
              action_idx, action_force, reward, done,
              dx_real_xdot, dx_real_thetadot,
              dx_nom_xdot, dx_nom_thetadot,
              dx_res_0, dx_res_1,
              nominal_error, residual_error,
              dhat_0, dhat_1, dhat_norm,
              uncertainty_0, uncertainty_1, uncertainty_mag
    """
    if cfg is None:
        cfg = DOBMBRLConfig()

    np.random.seed(run_idx)
    torch.manual_seed(run_idx)

    # --- 모델 로드 ---
    checkpoint_path = os.path.join(checkpoint_dir, f'Champion_Seed{run_idx}_BestModel.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'체크포인트 없음: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, weights_only=False)

    num_observations = 4
    num_actions      = 2
    num_act_features = 1

    q_network    = QNetwork(num_observations, num_actions)
    res_net      = ResidualDxNet(num_observations, num_act_features, hidden=32)
    uncert_model = NormalizedRBFModel(cfg.num_rbf_centers, cfg.rbf_width, cfg.rbf_initial_value)

    q_network.load_state_dict(ckpt['q_network'])
    res_net.load_state_dict(ckpt['res_net'])
    uncert_model.load_state_dict(ckpt['uncert_model'])

    q_network.eval()
    res_net.eval()
    uncert_model.eval()

    # --- 환경 & 파라미터 ---
    env    = make_cartpole_env()
    p_nom  = default_cartpole_params()

    step_logs = []

    for episode_ct in range(1, num_test_episodes + 1):

        obs  = np.array(reset_env(env), dtype=np.float32)
        obs_t = torch.tensor(obs).unsqueeze(0)
        dhat = np.zeros(2, dtype=np.float32)   # 에피소드마다 리셋

        for step_ct in range(1, cfg.max_steps_per_ep + 1):

            # Greedy 행동 선택 (탐색 없음)
            with torch.no_grad():
                action_idx = int(q_network(obs_t).argmax(dim=1).item())

            action_force = float(ACT_ELEMENTS[action_idx])
            gym_action   = action_idx

            next_obs_raw, _, is_done, _ = step_env(env, gym_action)
            next_obs = np.array(next_obs_raw, dtype=np.float32)

            # Nominal dynamics
            x_nom_next = step_nominal_cartpole(
                obs.reshape(1, -1),
                np.array([[action_force]], dtype=np.float32),
                p_nom
            ).flatten()
            dx_nom  = x_nom_next - obs           # (4,)
            dx_real = next_obs - obs             # (4,)

            # ResidualDxNet 예측
            with torch.no_grad():
                inp_res = torch.tensor(
                    np.concatenate([obs, [action_force]], dtype=np.float32)
                ).unsqueeze(0)
                dx_res = res_net(inp_res).cpu().numpy().flatten()  # (2,)

            # DOB 업데이트 (trainer.py와 동일 로직)
            e    = dx_real - dx_nom
            dhat = cfg.dob_w * dx_res + (1.0 - cfg.dob_w) * (FPINV @ e)
            uncertainty = FPINV @ e - dx_res     # (2,)

            # 오차 계산
            nominal_error   = float(np.linalg.norm(e))
            residual_error  = float(np.linalg.norm(e - F_MAT @ dx_res))

            # reward
            reward_arr, _ = reward_is_done_function(next_obs.reshape(1, -1))
            reward = float(reward_arr[0])

            # FPINV 투영 (DOB가 보는 2성분만 기록)
            dx_real_proj = FPINV @ dx_real   # (2,)
            dx_nom_proj  = FPINV @ dx_nom    # (2,)

            step_logs.append({
                'seed'               : run_idx,
                'episode'            : episode_ct,
                'step'               : step_ct,
                # 관측
                'obs_x'              : float(obs[0]),
                'obs_xdot'           : float(obs[1]),
                'obs_theta'          : float(obs[2]),
                'obs_thetadot'       : float(obs[3]),
                # 행동
                'action_idx'         : action_idx,
                'action_force'       : action_force,
                # 환경 반응
                'reward'             : reward,
                'done'               : int(is_done),
                # 동역학 분해 (FPINV 투영 2성분)
                'dx_real_xdot'       : float(dx_real_proj[0]),
                'dx_real_thetadot'   : float(dx_real_proj[1]),
                'dx_nom_xdot'        : float(dx_nom_proj[0]),
                'dx_nom_thetadot'    : float(dx_nom_proj[1]),
                'dx_res_0'           : float(dx_res[0]),
                'dx_res_1'           : float(dx_res[1]),
                # 오차
                'nominal_error'      : nominal_error,
                'residual_error'     : residual_error,
                # DOB
                'dhat_0'             : float(dhat[0]),
                'dhat_1'             : float(dhat[1]),
                'dhat_norm'          : float(np.linalg.norm(dhat)),
                # 불확실성
                'uncertainty_0'      : float(uncertainty[0]),
                'uncertainty_1'      : float(uncertainty[1]),
                'uncertainty_mag'    : float(np.linalg.norm(uncertainty)),
            })

            obs   = next_obs
            obs_t = torch.tensor(obs).unsqueeze(0)

            if is_done:
                break

    env.close()
    return step_logs
