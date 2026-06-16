"""
model_learning.py — Ensemble model 학습 유틸리티

format_samples_for_training : buffer 샘플 → (inputs, targets) numpy 변환
train_ensemble              : EnsembleModel 학습 호출 래퍼
"""
import numpy as np
import torch

from ..models.ensemble_model import EnsembleModel
from ..utils.buffer import ReplayBuffer


def format_samples_for_training(real_buffer: ReplayBuffer):
    """
    real_buffer 전체 데이터를 Ensemble 학습용 입출력으로 변환.
    inputs  : (N, obs_dim + act_dim)       — [obs, act]
    targets : (N, rew_dim + obs_dim)       — [reward, delta_obs]
    """
    obs      = real_buffer.obs[:real_buffer.length]
    act      = real_buffer.act[:real_buffer.length]
    next_obs = real_buffer.next_obs[:real_buffer.length]
    rew      = real_buffer.rew[:real_buffer.length]

    delta_obs = next_obs - obs
    inputs    = np.concatenate([obs, act],          axis=-1).astype(np.float32)
    targets   = np.concatenate([rew[:, None], delta_obs], axis=-1).astype(np.float32)
    return inputs, targets


def train_ensemble(model: EnsembleModel, real_buffer: ReplayBuffer,
                   batch_size: int = 256,
                   holdout_ratio: float = 0.2,
                   max_epochs_since_update: int = 5,
                   max_t: float = None,
                   device: torch.device = None) -> dict:
    """
    Ensemble 모델을 real_buffer 전체 데이터로 학습.
    Returns model_metrics dict with 'val_loss'.
    """
    inputs, targets = format_samples_for_training(real_buffer)
    print(f'[ Ensemble ] Training | Buffer size: {real_buffer.length} '
          f'| Input: {inputs.shape} | Target: {targets.shape}')
    metrics = model.train_ensemble(
        inputs                = inputs,
        targets               = targets,
        batch_size            = batch_size,
        max_epochs_since_update = max_epochs_since_update,
        holdout_ratio         = holdout_ratio,
        max_t                 = max_t,
        device                = device,
    )
    return metrics
