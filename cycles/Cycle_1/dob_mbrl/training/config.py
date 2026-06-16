"""
config.py — DOB-MBRL 하이퍼파라미터 (Gymnasium Hopper-v5, SAC)
수정은 이 파일을 직접 편집. CLI override 없음.
"""
from dataclasses import dataclass


@dataclass
class DOBMBRLConfig:
    # 학습 기본
    total_steps: int         = 125000   # 총 환경 스텝 수 (학습 종료 기준)
    max_steps_per_ep: int    = 1000       # Hopper-v5 최대 스텝
    warm_start_steps: int    = 5000       # 무작위 탐색 구간 (step 기준)

    # SAC Critic
    lr_critic: float         = 3e-4
    discount_factor: float   = 0.99
    tau: float               = 0.005      # soft-update coefficient
    update_interval: int     = 1          # critic 업데이트 주기 (env step 단위)
    num_gradient_steps: int  = 10         # gradient steps per env step (20→10: Q-divergence 방지)

    # SAC Actor
    lr_actor: float          = 3e-4

    # SAC Temperature (entropy regularization)
    init_alpha: float        = 1.0        # log_alpha init=0.0, matches original MBPO
    lr_alpha: float          = 3e-4
    auto_tune_alpha: bool    = True       # True → alpha 자동 학습
    target_entropy: float    = -1.0       # Hopper 원본 MBPO 설정

    # Buffer
    buffer_size: int         = int(1e6)
    mini_batch_size: int     = 256
    num_epochs: int          = 5          # residual model training epochs

    # Model mixing
    real_ratio: float        = 0.05       # sample_mixed_minibatch real 비율 (원본 MBPO)

    # DOB
    dob_w: float             = 0.1

    # RBF
    num_rbf_centers: int     = 3000
    rbf_width: float         = 0.5
    rbf_initial_value: float = 5.0
    lr_rbf: float            = 0.5
    lr_residual: float       = 1e-3

    # Rollout
    uncertainty_threshold: float         = 2.0
    num_generate_sample_iteration: int   = 1
    epoch_length: int                    = 1000       # 1 epoch = 1000 env steps
    rollout_interval: int                = 250        # model training + rollout 주기 (env step 단위)
    rollout_batch_size: int              = int(100e3) # rollout 시작 obs 수 (원본 MBPO)
    rollout_horizon: int                 = 15         # 고정 horizon (uncertainty truncation이 OOD 처리)
    model_retain_epochs: int             = 1

    # Evaluation
    eval_interval: int                   = 50
    eval_episodes: int                   = 10

    # Ablation
    use_uncertainty_sampling: bool       = False
    use_rbf: bool                        = True

    def __post_init__(self):
        assert self.total_steps > 0, f"total_steps must be positive, got {self.total_steps}"
        assert self.warm_start_steps >= 0, f"warm_start_steps must be >= 0, got {self.warm_start_steps}"
        assert self.mini_batch_size <= self.buffer_size, (
            f"mini_batch_size ({self.mini_batch_size}) must be <= buffer_size ({self.buffer_size})"
        )
        assert 0.0 < self.real_ratio <= 1.0, (
            f"real_ratio must be in (0, 1], got {self.real_ratio}"
        )
        assert 0.0 < self.tau <= 1.0, (
            f"tau must be in (0, 1], got {self.tau}"
        )
        assert self.rollout_horizon >= 1, f"rollout_horizon must be >= 1, got {self.rollout_horizon}"
