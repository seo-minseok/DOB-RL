"""
eval_rollout_uncert_cartpole.py — 실제 CartPole 에피소드의 매 스텝마다
현재 상태에서 model-based rollout을 수행하고, 그 rollout 궤적과 uncertainty를
세 패널로 시각화하는 비디오를 생성한다.

Layout
------
Left   : Uncertainty heatmap (x vs theta, 현재 xdot/thetadot/action 조건부)
         + 현재 상태에서 출발한 NUM_ROLLOUTS 개 rollout 궤적
Center : Rollout 스텝별 uncertainty magnitude (임계값 점선 표시)
Right  : CartPole-v1 실제 환경 렌더링

사용법
------
    cd cycles/Cycle_1
    python eval_rollout_uncert_cartpole.py
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from dob_mbrl.models import QNetwork, ResidualDxNet, NormalizedRBFModel
from dob_mbrl.dynamics.constants import ACT_ELEMENTS, X_THRESHOLD, THETA_THRESHOLD
from dob_mbrl.dynamics.nominal import default_cartpole_params
from dob_mbrl.dynamics.dob import predict_next_obs_dob
from dob_mbrl.envs.cartpole_utils import reset_env, step_env


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
CHECKPOINT_DIR   = Path("checkpoints/baseline")
SEED             = 1
VIDEO_DIR        = Path("figures/eval_rollout_uncert_cartpole")
VIDEO_FILE_NAME  = "eval_rollout_uncert_cartpole.mp4"

MAX_STEPS         = 500    # 실제 에피소드 최대 스텝 수
THRESHOLD         = 0.1    # 체크포인트 학습 기준 uncertainty 임계값
FPS               = 10
FINAL_HOLD_FRAMES = 15

# 롤아웃 설정
NUM_ROLLOUTS       = 10
MAX_ROLLOUT_HORIZON = 20
HERO_IDX           = 0     # 노이즈 없는 기준 롤아웃 인덱스
NOISE_SCALE        = np.array([0.02, 0.02, 0.005, 0.05], dtype=np.float32)

# 히트맵 그리드 해상도
N_GRID   = 50
X_RANGE  = (-X_THRESHOLD * 1.05, X_THRESHOLD * 1.05)
TH_RANGE = (-THETA_THRESHOLD * 1.05, THETA_THRESHOLD * 1.05)

FIGURE_DPI = 100


# ---------------------------------------------------------------------------
# 체크포인트 로드
# ---------------------------------------------------------------------------
def resolve_checkpoint(checkpoint_dir: Path, seed: int) -> Path:
    path = checkpoint_dir / f"Champion_Seed{seed}_BestModel.pt"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"체크포인트를 찾을 수 없습니다: '{path}'\n"
        f"seed={seed} 로 학습을 완료한 뒤 실행하세요."
    )


def load_models(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    policy       = QNetwork()
    res_net      = ResidualDxNet(hidden=32)
    uncert_model = NormalizedRBFModel()

    policy.load_state_dict(ckpt["q_network"])
    res_net.load_state_dict(ckpt["res_net"])
    uncert_model.load_state_dict(ckpt["uncert_model"])

    policy.eval()
    res_net.eval()
    uncert_model.eval()

    p_nom = default_cartpole_params()
    return policy, res_net, uncert_model, p_nom


# ---------------------------------------------------------------------------
# 추론 헬퍼
# ---------------------------------------------------------------------------
def get_action(policy: QNetwork, obs: np.ndarray) -> tuple[int, float]:
    obs_t = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        action_idx = int(policy(obs_t).argmax(dim=1).item())
    return action_idx, float(ACT_ELEMENTS[action_idx])


def compute_uncertainty_batch(uncert_model: NormalizedRBFModel,
                               obs_batch: np.ndarray,
                               act_batch: np.ndarray) -> np.ndarray:
    """
    obs_batch : (N, 4), act_batch : (N, 1)
    Returns uncert_mag : (N,)
    """
    inp = np.concatenate([obs_batch, act_batch], axis=1).astype(np.float32)
    x_t = torch.tensor(inp, dtype=torch.float32)
    with torch.no_grad():
        pred = uncert_model(x_t).cpu().numpy()   # (N, 2)
    return np.sqrt(np.sum(pred ** 2, axis=1))


def compute_uncert_map(uncert_model: NormalizedRBFModel,
                        curr_xdot: float,
                        curr_thetadot: float,
                        curr_act: float,
                        n_grid: int = N_GRID):
    """(x, theta) 2D 그리드 uncertainty 계산."""
    x_grid  = np.linspace(X_RANGE[0],  X_RANGE[1],  n_grid, dtype=np.float32)
    th_grid = np.linspace(TH_RANGE[0], TH_RANGE[1], n_grid, dtype=np.float32)
    x_mesh, th_mesh = np.meshgrid(x_grid, th_grid)
    grid_n = x_mesh.size

    batch_input = np.vstack([
        x_mesh.reshape(1, grid_n),
        np.full((1, grid_n), curr_xdot,     dtype=np.float32),
        th_mesh.reshape(1, grid_n),
        np.full((1, grid_n), curr_thetadot, dtype=np.float32),
        np.full((1, grid_n), curr_act,      dtype=np.float32),
    ]).T  # (grid_n, 5)

    x_t = torch.tensor(batch_input, dtype=torch.float32)
    with torch.no_grad():
        pred = uncert_model(x_t).cpu().numpy()
    z_grid = np.sqrt(np.sum(pred ** 2, axis=1)).reshape(n_grid, n_grid)
    return x_grid, th_grid, z_grid


def render_figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(height, width, 4)[..., :3].copy()


# ---------------------------------------------------------------------------
# 현재 상태에서 model-based rollout 수행
# ---------------------------------------------------------------------------
def run_rollouts(obs: np.ndarray,
                 policy: QNetwork,
                 res_net: ResidualDxNet,
                 uncert_model: NormalizedRBFModel,
                 p_nom: dict,
                 rng: np.random.Generator) -> dict:
    """
    obs (4,) 현재 실제 상태에서 NUM_ROLLOUTS 개의 model-based rollout 수행.

    Returns
    -------
    dict with keys:
      obs_history    : (4, MAX_ROLLOUT_HORIZON+1, NUM_ROLLOUTS)
      uncert_history : (MAX_ROLLOUT_HORIZON, NUM_ROLLOUTS)
      act_history    : (MAX_ROLLOUT_HORIZON, NUM_ROLLOUTS)
      stop_steps     : (NUM_ROLLOUTS,) — 실제 수행된 스텝 수
    """
    obs_history    = np.zeros((4, MAX_ROLLOUT_HORIZON + 1, NUM_ROLLOUTS), dtype=np.float32)
    uncert_history = np.zeros((MAX_ROLLOUT_HORIZON, NUM_ROLLOUTS), dtype=np.float32)
    act_history    = np.zeros((MAX_ROLLOUT_HORIZON, NUM_ROLLOUTS), dtype=np.float32)
    stop_steps     = np.full(NUM_ROLLOUTS, MAX_ROLLOUT_HORIZON, dtype=np.int32)

    noise = rng.standard_normal((NUM_ROLLOUTS, 4)).astype(np.float32) * NOISE_SCALE
    noise[HERO_IDX] = 0.0   # hero는 노이즈 없음

    init_obs = obs.reshape(1, 4) + noise   # (NUM_ROLLOUTS, 4)

    for r in range(NUM_ROLLOUTS):
        curr_obs = init_obs[r].copy()
        obs_history[:, 0, r] = curr_obs

        for h in range(MAX_ROLLOUT_HORIZON):
            _, act_force = get_action(policy, curr_obs)
            act_arr = np.array([[act_force]], dtype=np.float32)

            uncert_mag = float(compute_uncertainty_batch(
                uncert_model,
                curr_obs.reshape(1, 4),
                act_arr,
            )[0])

            uncert_history[h, r] = uncert_mag
            act_history[h, r]    = act_force

            # h > 0 에서 임계값 초과 시 롤아웃 중단 (C2_val_v2.py 동일 로직)
            if uncert_mag > THRESHOLD and h > 0:
                stop_steps[r] = h + 1
                break

            next_obs = predict_next_obs_dob(
                curr_obs.reshape(1, 4),
                act_arr,
                res_net,
                p_nom,
                use_nominal=True,
            )[0]
            curr_obs = next_obs.astype(np.float32, copy=False)
            obs_history[:, h + 1, r] = curr_obs

    return {
        "obs_history":    obs_history,
        "uncert_history": uncert_history,
        "act_history":    act_history,
        "stop_steps":     stop_steps,
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ckpt_path = resolve_checkpoint(CHECKPOINT_DIR, SEED)
    policy, res_net, uncert_model, p_nom = load_models(ckpt_path)
    print(f"모델 로드 완료 (seed={SEED}). 평가 시작...")

    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1", render_mode="rgb_array")
    except Exception:
        import gym
        env = gym.make("CartPole-v1")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    obs = reset_env(env)

    # ---- Figure 구성: 왼쪽 heatmap+rollout, 가운데 uncertainty, 오른쪽 render ----
    fig = plt.figure(figsize=(18, 5), dpi=FIGURE_DPI, facecolor="white")
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                             width_ratios=[1.1, 1.0, 0.9], wspace=0.40)
    ax_map    = fig.add_subplot(gs[0])
    ax_uncert = fig.add_subplot(gs[1])
    ax_render = fig.add_subplot(gs[2])
    ax_render.axis("off")

    video_path = VIDEO_DIR / f"seed{SEED}" / VIDEO_FILE_NAME
    video_path.parent.mkdir(parents=True, exist_ok=True)

    ep_reward  = 0.0
    done       = False
    step_count = 0

    # 첫 스텝에서만 컬러바를 추가하기 위한 플래그
    colorbar_added = False

    with imageio.get_writer(video_path, fps=FPS, quality=8) as writer:
        while not done and step_count < MAX_STEPS:
            x, xdot, theta, thetadot = (
                float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
            )

            # ---- 정책으로 행동 선택 ----
            action_idx, action_force = get_action(policy, obs)

            # ---- 현재 상태에서 model-based rollout ----
            rollout = run_rollouts(obs, policy, res_net, uncert_model, p_nom, rng)
            obs_hist    = rollout["obs_history"]     # (4, H+1, R)
            uncert_hist = rollout["uncert_history"]  # (H, R)
            stop_steps  = rollout["stop_steps"]      # (R,)

            # ---- 실제 환경 스텝 ----
            obs_next, rew, done, info = step_env(env, action_idx)
            ep_reward += float(rew)

            # ---- 렌더 프레임 취득 ----
            try:
                render_frame = env.render()
            except Exception:
                render_frame = None

            # ---- Uncertainty 히트맵 (hero 롤아웃 첫 스텝 조건부) ----
            x_grid, th_grid, z_grid = compute_uncert_map(
                uncert_model, xdot, thetadot, action_force, N_GRID
            )

            # ============================================================
            # 왼쪽: Uncertainty Map + Rollout 궤적
            # ============================================================
            ax_map.cla()
            ax_map.set_title(
                f"Rollout in State Space  (step={step_count + 1})",
                fontsize=10, fontweight="bold",
            )
            ax_map.set_xlabel("Cart Position  x  [m]")
            ax_map.set_ylabel("Pole Angle  θ  [rad]")
            ax_map.set_xlim(X_RANGE)
            ax_map.set_ylim(TH_RANGE)

            im = ax_map.imshow(
                z_grid,
                extent=[x_grid.min(), x_grid.max(), th_grid.min(), th_grid.max()],
                origin="lower",
                cmap="jet",
                vmin=0.0,
                vmax=THRESHOLD * 1.5,
                alpha=0.55,
                aspect="auto",
            )

            # 위험 영역 경계
            for xv in [X_THRESHOLD, -X_THRESHOLD]:
                ax_map.axvline(xv, color="white", linestyle="--",
                               linewidth=1.0, alpha=0.6)
            for tv in [THETA_THRESHOLD, -THETA_THRESHOLD]:
                ax_map.axhline(tv, color="white", linestyle="--",
                               linewidth=1.0, alpha=0.6)

            # 롤아웃 궤적 그리기
            for r in range(NUM_ROLLOUTS):
                n_pts    = int(stop_steps[r])
                xs       = obs_hist[0, :n_pts, r]
                thetas   = obs_hist[2, :n_pts, r]
                is_dead  = (stop_steps[r] < MAX_ROLLOUT_HORIZON)

                if r == HERO_IDX:
                    color     = np.array([0.0, 0.85, 0.35], dtype=np.float32)
                    linewidth = 2.5
                    ms        = 8
                    alpha     = 1.0
                else:
                    color     = np.array([0.75, 0.75, 0.75], dtype=np.float32)
                    linewidth = 1.0
                    ms        = 4
                    alpha     = 0.5

                rgba = (*color.tolist(), alpha)
                ax_map.plot(xs, thetas, ".-",
                            color=rgba, linewidth=linewidth, markersize=ms // 2)

                # 마지막 점
                if is_dead:
                    ax_map.plot(xs[-1], thetas[-1], "rx",
                                markersize=ms + 2, linewidth=2, zorder=5)
                else:
                    ax_map.plot(xs[-1], thetas[-1], "o",
                                markerfacecolor=color,
                                markeredgecolor="white",
                                markersize=ms, zorder=5)

            # 현재 실제 상태 표시
            ax_map.plot(x, theta, "*",
                        markersize=14, markerfacecolor="yellow",
                        markeredgecolor="black", markeredgewidth=1.0,
                        zorder=6, label="Real state")
            ax_map.legend(loc="upper right", fontsize=7,
                          framealpha=0.85, facecolor="white")

            # 조건부 정보 텍스트
            hero_text = (
                f"xdot={xdot:+.2f}  θdot={thetadot:+.2f}\n"
                f"Action={action_force:+.0f} N"
            )
            ax_map.text(
                X_RANGE[0] + 0.05, TH_RANGE[1] - 0.003,
                hero_text,
                fontsize=8, color="white", fontweight="bold", va="top",
                bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=2),
            )

            # 컬러바 (첫 스텝에만)
            if not colorbar_added:
                cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
                cbar.set_label("Uncertainty Magnitude", fontsize=7)
                cbar.ax.axhline(THRESHOLD, color="red", linewidth=1.5)
                colorbar_added = True

            # ============================================================
            # 가운데: Rollout 스텝별 Uncertainty
            # ============================================================
            ax_uncert.cla()
            ax_uncert.set_title(
                "Rollout Uncertainty (C2 Criterion)",
                fontsize=10, fontweight="bold",
            )
            ax_uncert.set_xlabel("Rollout Step")
            ax_uncert.set_ylabel("Uncertainty Magnitude")
            ax_uncert.set_xlim(0, MAX_ROLLOUT_HORIZON)
            ax_uncert.set_ylim(0, THRESHOLD * 1.8)
            ax_uncert.grid(True, alpha=0.4)
            ax_uncert.axhline(THRESHOLD, color="red", linewidth=2,
                              linestyle="--", label=f"Threshold ({THRESHOLD})")

            for r in range(NUM_ROLLOUTS):
                n_pts   = int(stop_steps[r])
                is_dead = (stop_steps[r] < MAX_ROLLOUT_HORIZON)
                steps_x = np.arange(1, n_pts + 1)
                u_vals  = uncert_hist[:n_pts, r]

                if r == HERO_IDX:
                    color     = np.array([0.0, 0.447, 0.741], dtype=np.float32)
                    linewidth = 2.5
                    ms        = 8
                    alpha     = 1.0
                else:
                    color     = np.array([0.6, 0.6, 0.6], dtype=np.float32)
                    linewidth = 1.0
                    ms        = 4
                    alpha     = 0.4

                rgba = (*color.tolist(), alpha)
                ax_uncert.plot(steps_x, u_vals, ".-",
                               color=rgba, linewidth=linewidth)

                if is_dead:
                    ax_uncert.plot(steps_x[-1], u_vals[-1], "rx",
                                   markersize=ms + 4, linewidth=2, zorder=5)

            ax_uncert.legend(fontsize=8, loc="upper left")

            # ============================================================
            # 오른쪽: CartPole 렌더
            # ============================================================
            ax_render.cla()
            ax_render.axis("off")
            ax_render.set_title(
                f"CartPole-v1  (seed={SEED})",
                fontsize=10, fontweight="bold",
            )
            if render_frame is not None:
                ax_render.imshow(render_frame, aspect="auto")

            done_str = "  [DONE]" if done else ""
            ax_render.set_xlabel(
                f"x={x:+.3f}  θ={theta:+.4f}  R={ep_reward:.1f}{done_str}",
                fontsize=8,
            )

            fig.suptitle(
                f"DOB-MBRL Rollout Evaluation  |  Real Step {step_count + 1}",
                fontsize=12, fontweight="bold", y=1.01,
            )
            fig.tight_layout()

            # ---- 프레임 저장 ----
            frame = render_figure_to_rgb(fig)
            if done:
                for _ in range(FINAL_HOLD_FRAMES):
                    writer.append_data(frame)
            else:
                writer.append_data(frame)

            obs = obs_next
            step_count += 1

    plt.close(fig)
    env.close()
    print(f"평가 완료: {step_count} 스텝  |  총 보상: {ep_reward:.2f}")
    print(f"비디오 저장 완료: {video_path}")


if __name__ == "__main__":
    main()
