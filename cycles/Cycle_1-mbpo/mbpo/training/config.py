"""
config.py — MBPO 하이퍼파라미터 (Gymnasium Hopper-v5, SAC + Ensemble)
수정은 이 파일을 직접 편집. CLI override 없음.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class MBPOConfig:
    # 학습 기본
    total_steps: int          = 125000     # 총 환경 스텝 수
    max_steps_per_ep: int     = 1000         # Hopper-v5 최대 스텝
    warm_start_steps: int     = 5000         # 무작위 탐색 구간
    epoch_length: int         = 1000         # 1 epoch = 1000 env steps

    # SAC
    lr: float                 = 3e-4
    discount_factor: float    = 0.99
    tau: float                = 5e-3
    reward_scale: float       = 1.0
    n_train_repeat: int       = 20           # gradient steps per env step

    # SAC Temperature
    init_alpha: float         = 1.0          # log_alpha init=0.0, matches original MBPO
    lr_alpha: float           = 3e-4
    auto_tune_alpha: bool     = True
    target_entropy: float     = -1.0         # Hopper 원본 설정 (not 'auto'=-3)

    # Buffer
    buffer_size: int          = int(1e6)
    mini_batch_size: int      = 256

    # Ensemble Model
    num_networks: int          = 7
    num_elites: int           = 5
    hidden_dim: int           = 200          # 4 hidden layers × 200 nodes
    model_lr: float           = 1e-3
    model_batch_size: int     = 256
    holdout_ratio: float      = 0.2
    max_epochs_since_update: int = 5

    # Model Rollout
    rollout_interval: int     = 250          # model train + rollout 주기 (steps)
    rollout_batch_size: int   = int(100e3)   # rollout 시작 obs 수
    rollout_schedule: List[int] = field(     # [min_epoch, max_epoch, min_horizon, max_horizon]
        default_factory=lambda: [20, 150, 1, 15]
    )
    model_retain_epochs: int  = 1

    # Model mixing
    real_ratio: float         = 0.05         # sample_mixed_minibatch real 비율

    # Evaluation
    eval_interval: int        = 50           # 에피소드 단위
    eval_episodes: int        = 10

    def __post_init__(self):
        assert self.total_steps > 0
        assert self.warm_start_steps >= 0
        assert self.mini_batch_size <= self.buffer_size
        assert 0.0 < self.real_ratio <= 1.0
        assert 0.0 < self.tau <= 1.0
        assert self.num_elites <= self.num_networks
