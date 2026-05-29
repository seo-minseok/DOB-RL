"""
eval_uncert_cartpole.py — 실제 CartPole 환경에서 정책을 평가하면서
왼쪽에 현재 스텝의 Uncertainty Map, 오른쪽에 실제 환경 렌더링을 합성한 비디오를 생성한다.

Layout
------
Left  : Uncertainty heatmap (x vs theta), 현재 (xdot, thetadot, action) 조건부
Right : CartPole-v1 rgb_array 렌더 프레임

사용법
------
    cd cycles/Cycle_1
    python eval_uncert_cartpole.py
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
from dob_mbrl.envs.cartpole_utils import make_cartpole_env, reset_env, step_env


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
CHECKPOINT_DIR   = Path("checkpoints/baseline")
SEED             = 1
VIDEO_DIR        = Path("figures/eval_uncert_cartpole")
VIDEO_FILE_NAME  = "eval_uncert_cartpole.mp4"

MAX_STEPS        = 500          # 에피소드 최대 스텝 수
THRESHOLD        = 0.1          # 불확실성 임계값 (체크포인트 학습 기준)
FPS              = 10
FINAL_HOLD_FRAMES = 15

# 히트맵 그리드 해상도
N_GRID = 50
X_RANGE  = (-X_THRESHOLD * 1.05, X_THRESHOLD * 1.05)
TH_RANGE = (-THETA_THRESHOLD * 1.05, THETA_THRESHOLD * 1.05)

# 환경 렌더 크기를 figure에 맞추기 위한 DPI
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
    """정책으로부터 (action_idx, action_force) 반환."""
    obs_t = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        action_idx = int(policy(obs_t).argmax(dim=1).item())
    return action_idx, float(ACT_ELEMENTS[action_idx])


def compute_uncert_map(
    uncert_model: NormalizedRBFModel,
    curr_xdot: float,
    curr_thetadot: float,
    curr_act: float,
    n_grid: int = N_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (x, theta) 2D 그리드에서 불확실성 크기를 계산한다.
    curr_xdot, curr_thetadot, curr_act 는 현재 상태의 조건부 값.

    Returns
    -------
    x_grid  : (n_grid,)
    th_grid : (n_grid,)
    z_grid  : (n_grid, n_grid)  — uncertainty magnitude
    """
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
        pred = uncert_model(x_t).cpu().numpy()  # (grid_n, 2)

    uncert_mag = np.sqrt(np.sum(pred ** 2, axis=1))
    z_grid = uncert_mag.reshape(n_grid, n_grid)
    return x_grid, th_grid, z_grid


def render_figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(height, width, 4)[..., :3].copy()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ckpt_path = resolve_checkpoint(CHECKPOINT_DIR, SEED)
    policy, res_net, uncert_model, p_nom = load_models(ckpt_path)
    print(f"모델 로드 완료 (seed={SEED}). 평가 시작...")

    # rgb_array 렌더 모드로 환경 생성
    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1", render_mode="rgb_array")
    except Exception:
        import gym
        env = gym.make("CartPole-v1")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    obs = reset_env(env)

    # Figure 구성: 왼쪽 uncertainty map, 오른쪽 CartPole 렌더
    fig = plt.figure(figsize=(13, 5), dpi=FIGURE_DPI, facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.35)
    ax_uncert  = fig.add_subplot(gs[0])
    ax_render  = fig.add_subplot(gs[1])

    ax_render.axis("off")

    video_path = VIDEO_DIR / f"seed{SEED}" / VIDEO_FILE_NAME
    video_path.parent.mkdir(parents=True, exist_ok=True)

    ep_reward  = 0.0
    done       = False
    step_count = 0

    with imageio.get_writer(video_path, fps=FPS, quality=8) as writer:
        while not done and step_count < MAX_STEPS:
            # ---- 현재 상태 파싱 ----
            x, xdot, theta, thetadot = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])

            # ---- 정책으로 행동 선택 ----
            action_idx, action_force = get_action(policy, obs)

            # ---- 현재 상태에서의 실제 불확실성 (점) ----
            point_input = np.array([[x, xdot, theta, thetadot, action_force]], dtype=np.float32)
            x_t = torch.tensor(point_input, dtype=torch.float32)
            with torch.no_grad():
                point_pred = uncert_model(x_t).cpu().numpy()[0]
            curr_uncert_mag = float(np.sqrt(np.sum(point_pred ** 2)))

            # ---- 불확실성 맵 계산 ----
            x_grid, th_grid, z_grid = compute_uncert_map(
                uncert_model, xdot, thetadot, action_force, N_GRID
            )

            # ---- 환경 스텝 ----
            obs_next, rew, done, info = step_env(env, action_idx)
            ep_reward += float(rew)

            # ---- 렌더 프레임 취득 ----
            try:
                render_frame = env.render()
            except Exception:
                render_frame = None

            # ============================================================
            # 왼쪽: Uncertainty Map
            # ============================================================
            ax_uncert.cla()
            ax_uncert.set_title(
                f"Uncertainty Map  (step={step_count + 1})",
                fontsize=11, fontweight="bold",
            )
            ax_uncert.set_xlabel("Cart Position  x  [m]")
            ax_uncert.set_ylabel("Pole Angle  θ  [rad]")
            ax_uncert.set_xlim(X_RANGE)
            ax_uncert.set_ylim(TH_RANGE)

            im = ax_uncert.imshow(
                z_grid,
                extent=[x_grid.min(), x_grid.max(), th_grid.min(), th_grid.max()],
                origin="lower",
                cmap="jet",
                vmin=0.0,
                vmax=THRESHOLD * 1.5,
                alpha=0.75,
                aspect="auto",
            )

            # 위험 영역 경계 표시
            ax_uncert.axvline( X_THRESHOLD,  color="white", linestyle="--", linewidth=1.0, alpha=0.6)
            ax_uncert.axvline(-X_THRESHOLD,  color="white", linestyle="--", linewidth=1.0, alpha=0.6)
            ax_uncert.axhline( THETA_THRESHOLD,  color="white", linestyle="--", linewidth=1.0, alpha=0.6)
            ax_uncert.axhline(-THETA_THRESHOLD,  color="white", linestyle="--", linewidth=1.0, alpha=0.6)

            # 현재 상태 점 표시
            point_color = "red" if curr_uncert_mag > THRESHOLD else "lime"
            ax_uncert.plot(
                x, theta,
                "o",
                markersize=10,
                markeredgecolor="white",
                markerfacecolor=point_color,
                markeredgewidth=1.5,
                zorder=5,
                label=f"Current state\nU={curr_uncert_mag:.3f}",
            )
            ax_uncert.legend(
                loc="upper right",
                fontsize=8,
                framealpha=0.85,
                facecolor="white",
            )

            # 조건부 정보 텍스트
            info_text = (
                f"xdot={xdot:+.2f}  θdot={thetadot:+.2f}\n"
                f"Action={action_force:+.0f} N  Reward={ep_reward:.1f}"
            )
            ax_uncert.text(
                X_RANGE[0] + 0.05,
                TH_RANGE[1] - 0.003,
                info_text,
                fontsize=8,
                color="white",
                fontweight="bold",
                va="top",
                bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=2),
            )

            # 임계값 컬러바 (첫 스텝에만 추가)
            if step_count == 0:
                cbar = fig.colorbar(im, ax=ax_uncert, fraction=0.046, pad=0.04)
                cbar.set_label("Uncertainty Magnitude", fontsize=8)
                cbar.ax.axhline(THRESHOLD, color="red", linewidth=1.5)

            # ============================================================
            # 오른쪽: CartPole 렌더
            # ============================================================
            ax_render.cla()
            ax_render.axis("off")
            ax_render.set_title(
                f"CartPole-v1  (seed={SEED})",
                fontsize=11, fontweight="bold",
            )

            if render_frame is not None:
                ax_render.imshow(render_frame, aspect="auto")

            done_str = "  [DONE]" if done else ""
            ax_render.set_xlabel(
                f"x={x:+.3f}  θ={theta:+.4f}{done_str}",
                fontsize=9,
            )

            fig.suptitle(
                f"DOB-MBRL Evaluation  |  step {step_count + 1}",
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
