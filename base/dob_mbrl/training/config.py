"""
config.py — DOB-MBRL 하이퍼파라미터 (dataclass + __post_init__ 검증)
수정은 이 파일을 직접 편집. CLI override 없음.
"""
from dataclasses import dataclass


@dataclass
class DOBMBRLConfig:
    # 학습 기본
    num_episodes: int        = 200
    max_steps_per_ep: int    = 500
    warm_start_samples: int  = 200

    # Q-Network
    lr_critic: float         = 1e-3
    discount_factor: float   = 0.99
    tau: float               = 0.005          # soft-update coefficient
    update_interval: int     = 10
    num_gradient_steps: int  = 2

    # Exploration
    epsilon: float           = 1.0
    epsilon_min: float       = 0.01
    epsilon_decay: float     = 0.005

    # Buffer
    buffer_size: int         = int(1e5)
    mini_batch_size: int     = 256
    num_epochs: int          = 5              # residual model training epochs

    # Model mixing
    real_ratio: float        = 0.2            # sample_mixed_minibatch real 비율

    # DOB
    dob_w: float             = 0.1

    # RBF
    num_rbf_centers: int     = 600
    rbf_width: float         = 0.1
    rbf_initial_value: float = 5.0
    lr_rbf: float            = 0.5
    lr_residual: float       = 1e-2

    # Rollout
    max_horizon_length: int              = 10
    uncertainty_threshold: float         = 0.1
    num_generate_sample_iteration: int   = 20
    epsilon_min_model: float             = 0.1

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
        assert self.epsilon_min < self.epsilon, (
            f"epsilon_min must be < epsilon"
        )
