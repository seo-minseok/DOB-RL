"""
config.py — DOB-MBRL 하이퍼파라미터 (Gymnasium Hopper-v5, TD3)
수정은 이 파일을 직접 편집. CLI override 없음.
"""
from dataclasses import dataclass


@dataclass
class DOBMBRLConfig:
    # 학습 기본
    total_episodes: int      = 5000       # 총 에피소드 수 (학습 종료 기준)
    max_steps_per_ep: int    = 1000       # Hopper-v5 최대 스텝
    warm_start_episodes: int = 1000       # 무작위 탐색 구간 (에피소드 단위, Hopper 랜덤 에피소드 ~10스텝 → 1000ep ≈ 10000 steps)

    # TD3 Critic
    lr_critic: float         = 3e-4
    discount_factor: float   = 0.99
    tau: float               = 0.005      # soft-update coefficient
    update_interval: int     = 1          # critic 업데이트 주기 (env step 단위)
    num_gradient_steps: int  = 1          # update_interval 스텝당 gradient 업데이트 횟수

    # TD3 Actor
    lr_actor: float          = 3e-4
    policy_delay: int        = 2          # actor update 주기 (critic N회당 actor 1회)

    # TD3 탐색 노이즈
    expl_noise: float        = 0.1        # 환경 탐색 노이즈 std
    target_noise: float      = 0.2        # target policy smoothing noise std
    target_noise_clip: float = 0.5        # target noise clipping 범위

    # Buffer
    buffer_size: int         = int(1e6)
    mini_batch_size: int     = 256
    num_epochs: int          = 5          # residual model training epochs

    # Model mixing
    real_ratio: float        = 1.0        # sample_mixed_minibatch real 비율

    # DOB
    dob_w: float             = 0.1

    # RBF
    num_rbf_centers: int     = 1200
    rbf_width: float         = 0.3
    rbf_initial_value: float = 5.0
    lr_rbf: float            = 0.5
    lr_residual: float       = 1e-2

    # Rollout
    max_horizon_length: int              = 5
    uncertainty_threshold: float         = 0.1
    num_generate_sample_iteration: int   = 20
    epsilon_min_model: float             = 0.1   # model rollout 탐색 노이즈 std
    rollout_end_episode: int             = 5000  # horizon 선형 스케줄 종료 에피소드

    # Evaluation
    eval_interval: int                   = 50     # 평가 주기 (에피소드 단위)
    eval_episodes: int                   = 10     # 평가 에피소드 수

    # Ablation
    use_uncertainty_sampling: bool       = False   # False → uniform sampling

    def __post_init__(self):
        assert self.total_episodes > 0, f"total_episodes must be positive, got {self.total_episodes}"
        assert self.warm_start_episodes >= 0, f"warm_start_episodes must be >= 0, got {self.warm_start_episodes}"
        assert self.mini_batch_size <= self.buffer_size, (
            f"mini_batch_size ({self.mini_batch_size}) must be <= buffer_size ({self.buffer_size})"
        )
        assert 0.0 < self.real_ratio <= 1.0, (
            f"real_ratio must be in (0, 1], got {self.real_ratio}"
        )
        assert 0.0 < self.tau <= 1.0, (
            f"tau must be in (0, 1], got {self.tau}"
        )
