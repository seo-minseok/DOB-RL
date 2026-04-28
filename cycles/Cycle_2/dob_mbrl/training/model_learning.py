"""
model_learning.py — Residual Dynamics & Uncertainty RBF 모델 학습
Rollout 중에는 호출되지 않음 — Phase 1(에피소드 시작 전)에서만 수행.
BipedalWalker: FPINV는 (7, 24), dhat/uncertainty는 (N, 7).
"""
import numpy as np
import torch
import torch.nn as nn

from ..dynamics.constants import FPINV


def train_residual_dx_model_dob(res_net, optimizer,
                                 real_buffer, mini_batch_size: int,
                                 num_epochs: int) -> float:
    """
    uncertainty-weighted sampling으로 residual 모델 학습.
    Target: buffer의 dhat — DOB disturbance estimate (7D).
    """
    res_net.train()
    valid_len = real_buffer.length

    uncert_mag = np.linalg.norm(real_buffer.uncertainty[:valid_len], axis=1)  # (N,)
    weights    = uncert_mag + 1e-3
    probs      = weights / weights.sum()

    loss_sum = 0.0
    loss_ct  = 0

    for _ in range(num_epochs):
        num_iterations = valid_len // mini_batch_size
        for _ in range(num_iterations):
            idx    = np.random.choice(valid_len, size=mini_batch_size,
                                      replace=True, p=probs)
            obs_t  = torch.tensor(real_buffer.obs[idx])
            act_t  = torch.tensor(real_buffer.act[idx])
            dhat_t = torch.tensor(real_buffer.dhat[idx])   # (batch, 7)

            inp    = torch.cat([obs_t, act_t], dim=-1)
            dx_res = res_net(inp)                           # (batch, 7)
            loss   = nn.functional.mse_loss(dx_res, dhat_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            loss_ct  += 1

    return loss_sum / max(1, loss_ct)


def train_uncertainty_rbf(uncert_model, optimizer, real_buffer,
                           res_net, batch_size: int, epochs: int) -> float:
    """
    fresh_uncertainty = |FPINV * e - dxRes| 를 한 번 계산 후 반복 학습.
    FPINV: (7, 24), e: (N, 24) → fpinv_e: (N, 7)
    """
    valid_len = real_buffer.length
    if valid_len == 0:
        return float('nan')

    obs_all      = real_buffer.obs[:valid_len]
    act_all      = real_buffer.act[:valid_len]
    next_obs_all = real_buffer.next_obs[:valid_len]
    dx_nom_all   = real_buffer.dx_nom[:valid_len]

    with torch.no_grad():
        dl_in_all  = torch.tensor(np.concatenate([obs_all, act_all], axis=-1))
        dx_res_all = res_net(dl_in_all).cpu().numpy()    # (N, 7)

    dx_real_all  = next_obs_all - obs_all              # (N, 24)
    e_all        = dx_real_all - dx_nom_all            # (N, 24)
    fpinv_e      = e_all @ FPINV.T                     # (N, 7)
    fresh_uncert = fpinv_e - dx_res_all                # (N, 7)

    loss_sum = 0.0
    ct       = 0

    uncert_model.train()
    for _ in range(epochs):
        num_iter = valid_len // batch_size
        if num_iter == 0:
            break
        for _ in range(num_iter):
            idx      = np.random.randint(0, valid_len, size=batch_size)
            inp_t    = torch.tensor(
                np.concatenate([obs_all[idx], act_all[idx]], axis=-1))
            target_t = torch.tensor(np.abs(fresh_uncert[idx]))

            pred = uncert_model(inp_t)                   # (batch, 7)
            loss = nn.functional.mse_loss(pred, target_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            ct       += 1

    return loss_sum / max(1, ct)
