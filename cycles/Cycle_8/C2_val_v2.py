"""
C2_val_v2.py — Cycle 1 포트 (base/original/C2_val_v2.py → dob_mbrl 패키지 기준)

롤아웃 로직, 불확실성 계산, 히어로 조건부 히트맵, 비디오 렌더링 순서는
원본 MATLAB 스크립트에 맞춰 유지.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from dob_mbrl.models import QNetwork, ResidualDxNet, NormalizedRBFModel
from dob_mbrl.dynamics.constants import ACT_ELEMENTS
from dob_mbrl.dynamics.nominal import default_cartpole_params
from dob_mbrl.dynamics.dob import predict_next_obs_dob


# ---------------------------------------------------------------------------
# 체크포인트 설정 — Cycle 1 학습 결과 경로
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path("checkpoints/baseline")
SEED = 1                             # Champion_Seed{SEED}_BestModel.pt
VIDEO_DIR = Path("figures/c2_adaptive_rollout")  # figures/c2_adaptive_rollout/seed{SEED}/
VIDEO_FILE_NAME = "C2_Batch_Adaptive_Rollout_Hero.mp4"

# ---------------------------------------------------------------------------
# 롤아웃 파라미터 — Cycle 1 config.py 기준값과 일치
# ---------------------------------------------------------------------------
NUM_ROLLOUTS   = 10
OBS0_BASE      = np.array([0.0, 0.0, -0.1, 0.1], dtype=np.float32)
MAX_HORIZON    = 10                  # cfg.max_horizon_length = 10
THRESHOLD      = 0.1                 # cfg.uncertainty_threshold = 0.1
HERO_IDX       = 0
FPS            = 4
FINAL_HOLD_FRAMES = 15


def resolve_checkpoint_path(checkpoint_dir: Path, seed: int) -> Path:
    path = checkpoint_dir / f"Champion_Seed{seed}_BestModel.pt"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"체크포인트를 찾을 수 없습니다: '{path}'\n"
        f"먼저 seed={seed}로 학습을 완료한 뒤 실행하세요."
    )


def load_models(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Cycle 1 체크포인트에는 p_nom이 저장되지 않으므로 기본값 사용
    p_nom = default_cartpole_params()

    policy       = QNetwork()
    res_net      = ResidualDxNet(hidden=32)
    uncert_model = NormalizedRBFModel()

    policy.load_state_dict(checkpoint["q_network"])
    res_net.load_state_dict(checkpoint["res_net"])
    uncert_model.load_state_dict(checkpoint["uncert_model"])

    policy.eval()
    res_net.eval()
    uncert_model.eval()
    return policy, res_net, uncert_model, p_nom


def get_action(policy: QNetwork, obs: np.ndarray) -> np.float32:
    obs_batch = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        action_idx = int(policy(obs_batch).argmax(dim=1).item())
    return np.float32(ACT_ELEMENTS[action_idx])


def forward_normalized_rbf(model: NormalizedRBFModel, x: np.ndarray) -> np.ndarray:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        y = model(x_tensor).cpu().numpy()
    return y


def render_figure(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buffer.reshape(height, width, 4)[..., :3].copy()


def main():
    checkpoint_path = resolve_checkpoint_path(CHECKPOINT_DIR, SEED)
    policy, res_net, uncert_model, p_nom = load_models(checkpoint_path)
    print(f"모델 로드 완료 (seed={SEED}). 비디오 생성 시작...")

    rng = np.random.default_rng(42)
    noise_scale  = np.array([0.05, 0.05, 0.01, 0.1], dtype=np.float32).reshape(4, 1)
    obs0_batch   = np.repeat(OBS0_BASE.reshape(4, 1), NUM_ROLLOUTS, axis=1)
    obs0_batch  += rng.standard_normal((4, NUM_ROLLOUTS)).astype(np.float32) * noise_scale

    uncert_history = np.zeros((MAX_HORIZON, NUM_ROLLOUTS), dtype=np.float32)
    obs_history    = np.zeros((4, MAX_HORIZON + 1, NUM_ROLLOUTS), dtype=np.float32)
    act_history    = np.zeros((MAX_HORIZON, NUM_ROLLOUTS), dtype=np.float32)
    stop_steps     = np.full(NUM_ROLLOUTS, MAX_HORIZON, dtype=np.int32)

    for rollout_idx in range(NUM_ROLLOUTS):
        obs = obs0_batch[:, rollout_idx].copy()
        obs_history[:, 0, rollout_idx] = obs

        for step_idx in range(MAX_HORIZON):
            act = get_action(policy, obs)

            pred_uncert = forward_normalized_rbf(
                uncert_model,
                np.concatenate([obs, np.array([act], dtype=np.float32)]).reshape(1, -1),
            )[0]
            uncert_mag = float(np.sqrt(np.sum(pred_uncert ** 2)))

            uncert_history[step_idx, rollout_idx] = uncert_mag
            act_history[step_idx, rollout_idx]    = act

            if uncert_mag > THRESHOLD and step_idx > 0:
                stop_steps[rollout_idx] = step_idx + 1
                break

            next_obs = predict_next_obs_dob(
                obs.reshape(1, -1),
                np.array([[act]], dtype=np.float32),
                res_net,
                p_nom,
                True,
            )[0]
            obs = next_obs.astype(np.float32, copy=False)
            obs_history[:, step_idx + 1, rollout_idx] = obs

    max_global_step = int(stop_steps.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.suptitle("C2 Adaptive Rollout (Hero Focus)")

    ax1.set_xlabel("Cart Position (x)")
    ax1.set_ylabel("Pole Angle (theta)")
    ax1.set_title("2D State Space (Conditioned on Hero Trajectory)")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.2, 0.2)
    ax1.grid(True)

    ax2.set_xlabel("Rollout Step")
    ax2.set_ylabel("Uncertainty Magnitude")
    ax2.set_title("Truncation Criterion (C2)")
    ax2.set_xlim(0, MAX_HORIZON)
    ax2.set_ylim(0, THRESHOLD * 1.5)
    ax2.grid(True)
    ax2.axhline(THRESHOLD, color="r", linewidth=2)

    n_grid  = 50
    x_grid  = np.linspace(-1.5, 1.5, n_grid, dtype=np.float32)
    th_grid = np.linspace(-0.2, 0.2, n_grid, dtype=np.float32)
    x_mesh, th_mesh = np.meshgrid(x_grid, th_grid)
    grid_n = x_mesh.size

    video_path = VIDEO_DIR / f"seed{SEED}" / VIDEO_FILE_NAME
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=FPS, quality=10) as writer:
        for step in range(1, max_global_step + 1):
            ax1.cla()
            ax1.set_xlabel("Cart Position (x)")
            ax1.set_ylabel("Pole Angle (theta)")
            ax1.set_title("2D State Space (Conditioned on Hero Trajectory)")
            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-0.2, 0.2)
            ax1.grid(True)

            hero_step      = min(step, int(stop_steps[HERO_IDX]))
            hero_state_idx = hero_step - 1

            curr_vel = float(obs_history[1, hero_state_idx, HERO_IDX])
            curr_thd = float(obs_history[3, hero_state_idx, HERO_IDX])
            curr_act = float(act_history[min(hero_step, act_history.shape[0]) - 1, HERO_IDX])

            batch_input = np.vstack([
                x_mesh.reshape(1, grid_n),
                np.full((1, grid_n), curr_vel, dtype=np.float32),
                th_mesh.reshape(1, grid_n),
                np.full((1, grid_n), curr_thd, dtype=np.float32),
                np.full((1, grid_n), curr_act, dtype=np.float32),
            ]).T

            pred_grid   = forward_normalized_rbf(uncert_model, batch_input)
            grid_uncert = np.sqrt(np.sum(pred_grid ** 2, axis=1))
            z_grid      = grid_uncert.reshape(n_grid, n_grid)

            ax1.imshow(
                z_grid,
                extent=[x_grid.min(), x_grid.max(), th_grid.min(), th_grid.max()],
                origin="lower",
                cmap="jet",
                vmin=0.0,
                vmax=THRESHOLD * 1.5,
                alpha=0.55,
                aspect="auto",
            )

            hero_text = f"Step: {step}\nHero Cond: v={curr_vel:.2f}, omega={curr_thd:.2f}"
            ax1.text(
                -1.4, 0.16,
                hero_text,
                color="white",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="black"),
            )

            ax2.cla()
            ax2.set_xlabel("Rollout Step")
            ax2.set_ylabel("Uncertainty Magnitude")
            ax2.set_title("Truncation Criterion (C2)")
            ax2.set_xlim(0, MAX_HORIZON)
            ax2.set_ylim(0, THRESHOLD * 1.5)
            ax2.grid(True)
            ax2.axhline(THRESHOLD, color="r", linewidth=2)

            for rollout_idx in range(NUM_ROLLOUTS):
                end_step = min(step, int(stop_steps[rollout_idx]))
                is_dead  = step >= int(stop_steps[rollout_idx])

                if rollout_idx == HERO_IDX:
                    line_color   = np.array([0.0, 0.447, 0.741], dtype=np.float32)
                    line_width   = 2.5
                    marker_size  = 8
                    alpha_val    = 1.0
                else:
                    line_color   = np.array([0.6, 0.6, 0.6], dtype=np.float32)
                    line_width   = 1.0
                    marker_size  = 4
                    alpha_val    = 0.4

                rgba_color = tuple(np.append(line_color, alpha_val))
                xs     = obs_history[0, :end_step, rollout_idx]
                thetas = obs_history[2, :end_step, rollout_idx]
                ax1.plot(xs, thetas, ".-", color=rgba_color, linewidth=line_width)

                if is_dead:
                    ax1.plot(
                        obs_history[0, end_step - 1, rollout_idx],
                        obs_history[2, end_step - 1, rollout_idx],
                        "rx", markersize=marker_size + 2, linewidth=2,
                    )
                else:
                    ax1.plot(
                        obs_history[0, end_step - 1, rollout_idx],
                        obs_history[2, end_step - 1, rollout_idx],
                        "o",
                        markeredgecolor="k",
                        markerfacecolor=line_color,
                        markersize=marker_size,
                    )

                ax2.plot(
                    np.arange(1, end_step + 1),
                    uncert_history[:end_step, rollout_idx],
                    ".-", color=rgba_color, linewidth=line_width,
                )
                if is_dead:
                    ax2.plot(
                        end_step,
                        uncert_history[end_step - 1, rollout_idx],
                        "rx", markersize=marker_size + 4, linewidth=2,
                    )

            frame = render_figure(fig)
            if step == max_global_step:
                for _ in range(FINAL_HOLD_FRAMES):
                    writer.append_data(frame)
            else:
                writer.append_data(frame)

    plt.close(fig)
    print(f"비디오 저장 완료: {video_path}")


if __name__ == "__main__":
    main()
