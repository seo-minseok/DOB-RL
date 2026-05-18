"""
diagnose_action_deafness.py — Action-Deafness 진단 스크립트

체크포인트에서 res_net(+ actor, critic)을 로드하여 3가지 지표를 측정:
  1. 크기 비율: ||dx_nom|| vs ||F_MAT.T @ dx_res|| — nominal이 지배적인지 확인
  2. Action 민감도: 동일 obs에서 action 변화 → next_obs 변화량 std
  3. Jacobian: ∂dx_res/∂act 노름 — res_net이 action에 얼마나 민감한지

실행:
  cd c:/Users/seominseok/DOB-RL
  python -m cycles.Cycle_5.scripts.diagnose_action_deafness \
      --ckpt cycles/Cycle_5/checkpoints/real_ratio=0.2_uncert_thresh=0.25/Champion_Seed1_BestModel.pt
"""
import argparse
import sys
import os
import numpy as np
import torch

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from cycles.Cycle_5.dob_mbrl.models.residual_dx_net import ResidualDxNet
from cycles.Cycle_5.dob_mbrl.models.actor_network import ActorNetwork
from cycles.Cycle_5.dob_mbrl.models.q_network import QNetwork
from cycles.Cycle_5.dob_mbrl.dynamics.nominal import (
    step_nominal_bipedalwalker, default_bipedalwalker_params
)
from cycles.Cycle_5.dob_mbrl.dynamics.constants import (
    F_MAT, FPINV, OBS_DIM, ACT_DIM, DOB_DIM,
    OBS_DIM_NAMES, VELOCITY_INDICES,
)


SEPARATOR = "=" * 60


def load_models(ckpt_path: str):
    print(f"\n[로드] {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    res_net = ResidualDxNet(OBS_DIM, ACT_DIM, hidden=64)
    res_net.load_state_dict(ckpt['res_net'])
    res_net.eval()

    actor = ActorNetwork(OBS_DIM, ACT_DIM)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    critic1 = QNetwork(OBS_DIM, ACT_DIM)
    critic1.load_state_dict(ckpt['critic1'])
    critic1.eval()

    saved_ep = ckpt.get('episode', '?')
    saved_steps = ckpt.get('total_steps', '?')
    print(f"  → episode={saved_ep}, total_steps={saved_steps}")
    return res_net, actor, critic1


def make_test_obs(n: int, seed: int = 42) -> np.ndarray:
    """합리적인 범위의 랜덤 관측값 생성 (환경 실제값 근사)"""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-0.5, 0.5, (n, OBS_DIM)).astype(np.float32)
    obs[:, 8]  = (rng.random(n) > 0.5).astype(np.float32)   # left_contact
    obs[:, 13] = (rng.random(n) > 0.5).astype(np.float32)   # right_contact
    return obs


# ─── 진단 1: 크기 비율 ────────────────────────────────────────────
def diag_magnitude(res_net, obs: np.ndarray, p_nom: dict):
    print(f"\n{SEPARATOR}")
    print("진단 1: dx_nom vs dx_res 크기 비율")
    print(SEPARATOR)

    # nominal은 action 무관 → zero action으로 계산
    act_zero = np.zeros((len(obs), ACT_DIM), dtype=np.float32)
    x_nom_next = step_nominal_bipedalwalker(obs, act_zero, p_nom)
    dx_nom = x_nom_next - obs   # (N, 14)

    inp = torch.tensor(np.concatenate([obs, act_zero], axis=-1))
    with torch.no_grad():
        dx_res_7 = res_net(inp).cpu().numpy()   # (N, 7)
    dx_res_14 = dx_res_7 @ F_MAT.T             # (N, 14) — velocity 인덱스만

    nom_mag = np.linalg.norm(dx_nom, axis=1)           # (N,)
    res_mag = np.linalg.norm(dx_res_14, axis=1)        # (N,)

    # velocity 차원만 별도 비교 (nominal은 velocity를 변경하지 않음)
    nom_vel = np.linalg.norm(dx_nom[:, VELOCITY_INDICES], axis=1)   # 항상 ~0
    res_vel = np.linalg.norm(dx_res_7, axis=1)                      # dx_res 직접

    print(f"  N={len(obs)}개 상태 평균:")
    print(f"  ||dx_nom||        = {nom_mag.mean():.5f} ± {nom_mag.std():.5f}")
    print(f"  ||dx_res_14||     = {res_mag.mean():.5f} ± {res_mag.std():.5f}")
    print(f"  ||dx_nom_vel||    = {nom_vel.mean():.5f}  (항상 ~0, nominal은 velocity 안 바꿈)")
    print(f"  ||dx_res_7 (raw)  = {res_vel.mean():.5f} ± {res_vel.std():.5f}")
    ratio = nom_mag.mean() / (res_mag.mean() + 1e-8)
    print(f"\n  → 비율 ||dx_nom|| / ||dx_res_14|| = {ratio:.2f}x")
    if ratio > 5.0:
        print("  [경고] Nominal이 Residual보다 5배 이상 큼 → Action 효과가 미미함")
    elif ratio > 2.0:
        print("  [주의] Nominal이 Residual보다 2~5배 큼 → 부분적 Action 무력화 가능")
    else:
        print("  [양호] Residual이 Nominal과 유사한 크기")

    # 차원별 분석
    print(f"\n  차원별 dx_nom (position 인덱스만 영향):")
    for i, name in enumerate(OBS_DIM_NAMES):
        tag = " ← action 무관(nominal)" if i not in VELOCITY_INDICES else " ← res_net만"
        print(f"    [{i:2d}] {name:16s}: nom={abs(dx_nom[:, i]).mean():.5f}  res={abs(dx_res_14[:, i]).mean():.5f}{tag}")

    return ratio


# ─── 진단 2: Action 민감도 ───────────────────────────────────────
def diag_action_sensitivity(res_net, obs: np.ndarray, p_nom: dict,
                             n_action_samples: int = 50):
    print(f"\n{SEPARATOR}")
    print("진단 2: Action 민감도 (동일 obs에서 action 변화 → next_obs 변화)")
    print(SEPARATOR)

    rng = np.random.default_rng(1234)
    n_states = min(50, len(obs))
    obs_fixed = obs[:n_states]

    next_obs_list = []
    action_list = []
    for _ in range(n_action_samples):
        act = rng.uniform(-1.0, 1.0, (n_states, ACT_DIM)).astype(np.float32)
        action_list.append(act)
        x_nom_next = step_nominal_bipedalwalker(obs_fixed, act, p_nom)
        dx_nom = x_nom_next - obs_fixed   # (n_states, 14)
        inp = torch.tensor(np.concatenate([obs_fixed, act], axis=-1))
        with torch.no_grad():
            dx_res_7 = res_net(inp).cpu().numpy()
        next_obs = obs_fixed + dx_nom + dx_res_7 @ F_MAT.T
        next_obs_list.append(next_obs)

    next_obs_arr = np.stack(next_obs_list, axis=0)   # (n_samples, n_states, 14)

    # 각 state에서 action 다양성에 따른 next_obs의 std
    std_per_dim = next_obs_arr.std(axis=0)   # (n_states, 14)
    std_mean = std_per_dim.mean(axis=0)       # (14,) 차원별 평균 std

    # 차원별로 nominal vs residual 기여 구분
    vel_idx_set = set(VELOCITY_INDICES.tolist())

    print(f"  {n_action_samples}가지 랜덤 action에 대한 next_obs의 표준편차:")
    print(f"  (값이 작을수록 action 변화가 next_obs에 영향 없음 → Action-Deafness)")
    print()
    total_vel_std = 0.0
    total_pos_std = 0.0
    for i, name in enumerate(OBS_DIM_NAMES):
        layer = "vel(res_net)" if i in vel_idx_set else "pos(nominal)"
        print(f"    [{i:2d}] {name:16s} [{layer}]: std={std_mean[i]:.6f}")
        if i in vel_idx_set:
            total_vel_std += std_mean[i]
        else:
            total_pos_std += std_mean[i]

    print(f"\n  velocity 차원 합산 std: {total_vel_std:.6f}  ← action이 res_net 통해 변화시키는 부분")
    print(f"  position 차원 합산 std: {total_pos_std:.6f}  ← 항상 ~0 (nominal은 action 무관)")

    if total_vel_std < 0.001:
        print("\n  [심각] Velocity 차원도 action에 거의 반응 없음 → 심각한 Action-Deafness")
    elif total_vel_std < 0.01:
        print("\n  [경고] Velocity 차원 action 반응 매우 작음")
    else:
        print("\n  [양호] Velocity 차원에서 action에 의미 있는 반응 있음")

    return std_mean


# ─── 진단 3: Jacobian (∂dx_res/∂act) ────────────────────────────
def diag_jacobian(res_net, obs: np.ndarray):
    print(f"\n{SEPARATOR}")
    print("진단 3: Jacobian ∂dx_res/∂act 노름")
    print(SEPARATOR)
    print("  (클수록 res_net이 action 변화에 민감하게 반응)")
    print()

    rng = np.random.default_rng(99)
    n_states = min(100, len(obs))
    obs_t = torch.tensor(obs[:n_states])

    jac_norms = []
    for _ in range(20):   # 20가지 랜덤 action
        act_np = rng.uniform(-1.0, 1.0, (n_states, ACT_DIM)).astype(np.float32)
        act_t = torch.tensor(act_np, requires_grad=True)
        inp = torch.cat([obs_t, act_t], dim=-1)
        dx_res = res_net(inp)   # (n_states, 7)
        # 각 state에서 scalar loss = dx_res.sum()으로 Jacobian 근사
        loss = dx_res.sum()
        loss.backward()
        jac_norm = act_t.grad.norm(dim=1).detach().numpy()   # (n_states,)
        jac_norms.append(jac_norm)
        act_t.grad = None

    jac_arr = np.concatenate(jac_norms)   # (n_states*20,)
    print(f"  ||∂dx_res/∂act|| (평균) = {jac_arr.mean():.6f}")
    print(f"  ||∂dx_res/∂act|| (최대) = {jac_arr.max():.6f}")
    print(f"  ||∂dx_res/∂act|| (최소) = {jac_arr.min():.6f}")

    if jac_arr.mean() < 0.01:
        print("\n  [심각] res_net이 action에 거의 반응하지 않음 → Action-Deafness 확인")
    elif jac_arr.mean() < 0.1:
        print("\n  [경고] res_net의 action 반응이 약함")
    else:
        print("\n  [양호] res_net이 action에 민감하게 반응")

    return jac_arr.mean()


# ─── 진단 4: Actor 출력 분포 ─────────────────────────────────────
def diag_actor_output(actor, obs: np.ndarray):
    print(f"\n{SEPARATOR}")
    print("진단 4: Actor 출력 분포 (학습된 정책이 의미 있는 행동을 출력하는가)")
    print(SEPARATOR)

    obs_t = torch.tensor(obs)
    with torch.no_grad():
        actions = actor(obs_t).cpu().numpy()   # (N, 4)

    print(f"  행동 차원별 통계 (각 차원 ∈ [-1, 1]):")
    for j in range(ACT_DIM):
        a = actions[:, j]
        print(f"    act[{j}]: mean={a.mean():.4f}, std={a.std():.4f}, "
              f"min={a.min():.4f}, max={a.max():.4f}")

    overall_std = actions.std(axis=0).mean()
    print(f"\n  평균 std (4차원 평균) = {overall_std:.4f}")
    if overall_std < 0.05:
        print("  [경고] Actor가 거의 동일한 행동만 출력 → 정책 붕괴(policy collapse) 가능")
    else:
        print("  [양호] Actor 출력 다양성 있음")


# ─── 진단 5: Q 기울기 크기 (∂Q/∂act) ────────────────────────────
def diag_q_gradient(actor, critic1, obs: np.ndarray):
    print(f"\n{SEPARATOR}")
    print("진단 5: Q 기울기 ||∂Q/∂act|| — Q-function이 action에 얼마나 민감한가")
    print(SEPARATOR)

    obs_t = torch.tensor(obs[:100])
    act_t = torch.zeros(len(obs_t), ACT_DIM, requires_grad=True)
    q_val = critic1(obs_t, act_t)   # (n, 1)
    q_val.sum().backward()
    grad_norm = act_t.grad.norm(dim=1).detach().numpy()

    print(f"  ||∂Q/∂act|| (평균) = {grad_norm.mean():.6f}")
    print(f"  ||∂Q/∂act|| (최대) = {grad_norm.max():.6f}")
    if grad_norm.mean() < 0.01:
        print("  [경고] Q가 action에 거의 민감하지 않음 → Actor가 Q gradient를 통해 학습 불가")
    else:
        print("  [양호] Q가 action에 의미 있는 기울기 제공")


# ─── 메인 ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True,
                        help='체크포인트 .pt 파일 경로')
    parser.add_argument('--n_states', type=int, default=500,
                        help='테스트에 사용할 랜덤 상태 수 (기본 500)')
    args = parser.parse_args()

    res_net, actor, critic1 = load_models(args.ckpt)
    p_nom = default_bipedalwalker_params()
    obs = make_test_obs(args.n_states)

    print(f"\n{'='*60}")
    print("Action-Deafness 진단 보고서")
    print(f"{'='*60}")
    print(f"체크포인트: {args.ckpt}")
    print(f"테스트 상태 수: {args.n_states}")

    ratio    = diag_magnitude(res_net, obs, p_nom)
    std_mean = diag_action_sensitivity(res_net, obs, p_nom)
    jac_mean = diag_jacobian(res_net, obs)
    diag_actor_output(actor, obs)
    diag_q_gradient(actor, critic1, obs)

    print(f"\n{SEPARATOR}")
    print("종합 판정")
    print(SEPARATOR)
    vel_std_sum = std_mean[VELOCITY_INDICES].sum()
    signs = []
    if ratio > 5.0:
        signs.append(f"  ✗ dx_nom/dx_res 비율 {ratio:.1f}x (>5) → Nominal 지배")
    if vel_std_sum < 0.01:
        signs.append(f"  ✗ Velocity action-std {vel_std_sum:.5f} (<0.01) → 사실상 고정")
    if jac_mean < 0.05:
        signs.append(f"  ✗ Jacobian {jac_mean:.5f} (<0.05) → res_net action 둔감")

    if signs:
        print("  [Action-Deafness 징후 감지]")
        for s in signs:
            print(s)
    else:
        print("  [Action-Deafness 징후 없음 — 다른 원인 조사 필요]")

    print()


if __name__ == '__main__':
    main()
