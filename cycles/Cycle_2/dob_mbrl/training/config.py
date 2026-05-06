"""
config.py — DOB-MBRL BipedalWalker TD3 하이퍼파라미터
수정은 이 파일을 직접 편집. CLI override 없음.
"""
from dataclasses import dataclass


@dataclass
class DOBMBRLConfig:
    # 학습 기본
    num_episodes: int        = 2000
    max_steps_per_ep: int    = 1600   # BipedalWalker-v3 기본 max steps
    warm_start_samples: int  = 10000  # 무작위 탐색 후 학습 시작

    # TD3 Critic
    lr_critic: float         = 3e-4
    discount_factor: float   = 0.99
    tau: float               = 0.005          # soft-update coefficient
    update_interval: int     = 10
    num_gradient_steps: int  = 1              # UTD = 1

    # TD3 Actor
    lr_actor: float          = 3e-4
    policy_delay: int        = 2              # 액터는 크리틱의 1/2 빈도로 업데이트

    # TD3 Exploration / Target Policy Noise
    expl_noise: float        = 0.1            # 탐색 노이즈 std (환경 상호작용)
    policy_noise: float      = 0.2            # 타깃 정책 노이즈 std
    noise_clip: float        = 0.5            # 타깃 정책 노이즈 클리핑 범위

    # Buffer
    buffer_size: int         = int(1e6)
    mini_batch_size: int     = 256
    num_epochs: int          = 5              # residual model training epochs

    # Model mixing
    real_ratio: float        = 0.5            # 0.5 = 50% real / 50% model

    # DOB
    dob_w: float             = 0.1

    # RBF
    num_rbf_centers: int     = 2000   # 28D 입력 공간 커버리지 보정 (600 × 28/5 ≈ 3360 → 2000 타협)
    rbf_width: float         = 0.25   # 0.1 × sqrt(28/5) ≈ 0.237 → 0.25
    rbf_initial_value: float = 5.0
    lr_rbf: float            = 0.5
    lr_residual: float       = 1e-2

    # Rollout
    max_horizon_length: int              = 10
    uncertainty_threshold: float         = 0.7
    num_generate_sample_iteration: int   = 20
    epsilon_min_model: float             = 0.1   # model rollout 탐색 노이즈 std

    def __post_init__(self):
        assert self.mini_batch_size <= self.buffer_size, (
            f"mini_batch_size ({self.mini_batch_size}) must be <= buffer_size ({self.buffer_size})"
        )
        assert 0.0 < self.real_ratio <= 1.0, (
            f"real_ratio must be in (0, 1], got {self.real_ratio}"
        )
        assert 0.0 < self.tau <= 1.0, (
            f"tau must be in (0, 1], got {self.tau}"
        )
