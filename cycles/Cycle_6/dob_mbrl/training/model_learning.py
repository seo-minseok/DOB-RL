"""
model_learning.py — Residual Dynamics & Uncertainty RBF 모델 학습
Rollout 중에는 호출되지 않음 — Phase 1(에피소드 시작 전)에서만 수행.
BipedalWalker: FPINV는 (7, 14), dhat/uncertainty는 (N, 7).
"""
import numpy as np
import torch
import torch.nn as nn

from ..dynamics.constants import FPINV


def train_residual_dx_model_dob(res_net, optimizer,
                                 real_buffer, mini_batch_size: int,
                                 num_epochs: int):
    """
    균등 랜덤 샘플링으로 residual 모델 학습.
    Target: buffer의 dhat — DOB disturbance estimate (7D).

    Returns: (loss_avg, sampled_uncert_avg)
    """
    res_net.train()
    valid_len = real_buffer.length

    uncert_mag = np.linalg.norm(real_buffer.uncertainty[:valid_len], axis=1)  # (N,)

    loss_sum          = 0.0
    loss_ct           = 0
    sampled_uncert_sum = 0.0

    for _ in range(num_epochs):
        perm           = np.random.permutation(valid_len)
        num_iterations = valid_len // mini_batch_size
        for i in range(num_iterations):
            idx    = perm[i * mini_batch_size:(i + 1) * mini_batch_size]
            obs_t  = torch.tensor(real_buffer.obs[idx])
            act_t  = torch.tensor(real_buffer.act[idx])
            dhat_t = torch.tensor(real_buffer.dhat[idx])   # (batch, 7)

            inp    = torch.cat([obs_t, act_t], dim=-1)
            dx_res = res_net(inp)                           # (batch, 7)
            loss   = nn.functional.mse_loss(dx_res, dhat_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum           += loss.item()
            loss_ct            += 1
            sampled_uncert_sum += float(uncert_mag[idx].mean())

    loss_avg          = loss_sum / max(1, loss_ct)
    sampled_uncert_avg = sampled_uncert_sum / max(1, loss_ct)
    return loss_avg, sampled_uncert_avg


def train_uncertainty_rbf(uncert_model, optimizer, real_buffer, res_net,
                           batch_size: int, epochs: int) -> float:
    """
    타겟: 현재 res_net 기준으로 재계산한 |FPINV @ (dx_real - dx_nom) - res_net(obs, act)|.
    buffer 저장 시점의 uncertainty 대신 최신 res_net을 사용해 분포 이동 문제 방지.
    """
    valid_len = real_buffer.length
    if valid_len == 0:
        return float('nan')

    obs_all      = real_buffer.obs[:valid_len]
    act_all      = real_buffer.act[:valid_len]
    next_obs_all = real_buffer.next_obs[:valid_len]
    dx_nom_all   = real_buffer.dx_nom[:valid_len]

    loss_sum = 0.0
    ct       = 0

    uncert_model.train()
    for _ in range(epochs):
        num_iter = valid_len // batch_size
        if num_iter == 0:
            break
        perm = np.random.permutation(valid_len)
        for i in range(num_iter):
            idx     = perm[i * batch_size:(i + 1) * batch_size]
            obs_t   = torch.tensor(obs_all[idx])
            act_t   = torch.tensor(act_all[idx])
            nxt_t   = torch.tensor(next_obs_all[idx])
            dxnom_t = torch.tensor(dx_nom_all[idx])

            inp_t   = torch.cat([obs_t, act_t], dim=-1)

            with torch.no_grad():
                dx_res = res_net(inp_t)                                        # (batch, 7)

            e       = nxt_t - obs_t - dxnom_t                                 # (batch, 14)
            fpinv_e = torch.tensor(
                (e.numpy() @ FPINV.T).astype(np.float32))                     # (batch, 7)
            target_t = torch.abs(fpinv_e - dx_res)

            pred = uncert_model(inp_t)
            loss = nn.functional.mse_loss(pred, target_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            ct       += 1

    return loss_sum / max(1, ct)


def train_contact_net(contact_net, optimizer, real_buffer,
                      mini_batch_size: int, num_epochs: int) -> float:
    """
    ContactNet 학습 (BCE loss).
    Target: real_buffer.next_obs의 left_contact(8), right_contact(13).
    """
    contact_net.train()
    valid_len = real_buffer.length
    if valid_len == 0:
        return float('nan')

    loss_sum = 0.0
    ct       = 0

    for _ in range(num_epochs):
        perm           = np.random.permutation(valid_len)
        num_iterations = valid_len // mini_batch_size
        for i in range(num_iterations):
            idx      = perm[i * mini_batch_size:(i + 1) * mini_batch_size]
            obs_t    = torch.tensor(real_buffer.obs[idx])
            act_t    = torch.tensor(real_buffer.act[idx])
            nxt_t    = torch.tensor(real_buffer.next_obs[idx])

            target   = nxt_t[:, [8, 13]]                    # (batch, 2)
            inp      = torch.cat([obs_t, act_t], dim=-1)
            pred     = contact_net(inp)                      # (batch, 2), sigmoid
            loss     = nn.functional.binary_cross_entropy(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            ct       += 1

    return loss_sum / max(1, ct)


def evaluate_rbf_calibration(uncert_model, real_buffer,
                              sample_size: int = 4096):
    """
    real_buffer 샘플에서 RBF 예측 uncertainty와 실제 uncertainty를 비교한다.

    Returns
    -------
    calib_ratio : mean(||RBF pred||₂) / mean(||actual||₂)
                  1.0이면 스케일 일치, <1이면 under-predict, >1이면 over-predict
    calib_corr  : Pearson correlation(pred_mag, actual_mag)
                  1.0에 가까울수록 RBF가 고-uncertainty 상태를 올바르게 식별
    """
    valid_len = real_buffer.length
    if valid_len == 0:
        return float('nan'), float('nan')

    n   = min(valid_len, sample_size)
    idx = np.random.choice(valid_len, size=n, replace=False)

    obs           = real_buffer.obs[idx]
    act           = real_buffer.act[idx]
    actual_uncert = real_buffer.uncertainty[idx]   # (n, 7)

    uncert_model.eval()
    with torch.no_grad():
        inp  = torch.tensor(np.concatenate([obs, act], axis=-1))
        pred = uncert_model(inp).cpu().numpy()     # (n, 7)

    pred_mag   = np.linalg.norm(pred,           axis=1)   # (n,)
    actual_mag = np.linalg.norm(actual_uncert,  axis=1)   # (n,)

    calib_ratio = float(pred_mag.mean() / (actual_mag.mean() + 1e-8))

    if pred_mag.std() < 1e-8 or actual_mag.std() < 1e-8:
        calib_corr = float('nan')
    else:
        calib_corr = float(np.corrcoef(pred_mag, actual_mag)[0, 1])

    return calib_ratio, calib_corr
