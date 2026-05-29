"""
model_learning.py — Residual Dynamics & Uncertainty RBF 모델 학습
MATLAB: trainResidualDxModelDOB, trainUncertaintyRBF
Rollout 중에는 호출되지 않음 — Phase 1(에피소드 시작 전)에서만 수행.
"""
import numpy as np
import torch
import torch.nn as nn

from ..dynamics.constants import FPINV


def train_residual_dx_model_dob(res_net, optimizer,
                                 real_buffer, mini_batch_size: int,
                                 num_epochs: int,
                                 use_uncertainty_sampling: bool = True):
    """
    MATLAB: trainResidualDxModelDOB
    use_uncertainty_sampling=True : uncertainty-weighted sampling (baseline)
    use_uncertainty_sampling=False: uniform random sampling (ablation)
    Target: buffer의 dhat (DOB disturbance estimate).

    Returns: (loss_avg, sampled_uncert_avg)
      sampled_uncert_avg: 실제 샘플링된 데이터의 uncertainty magnitude 평균
    """
    res_net.train()
    valid_len = real_buffer.length

    uncert_mag = np.linalg.norm(real_buffer.uncertainty[:valid_len], axis=1)  # (N,)

    if use_uncertainty_sampling:
        # MATLAB: uncertMag = sqrt(sum(uncertainty.^2,1))
        weights = uncert_mag + 1e-3
        probs   = weights / weights.sum()
    else:
        probs = None  # uniform

    loss_sum         = 0.0
    loss_ct          = 0
    sampled_uncert_sum = 0.0
    sampled_uncert_ct  = 0

    for _ in range(num_epochs):
        num_iterations = valid_len // mini_batch_size
        for _ in range(num_iterations):
            if use_uncertainty_sampling:
                idx = np.random.choice(valid_len, size=mini_batch_size,
                                       replace=True, p=probs)
            else:
                idx = np.random.randint(0, valid_len, size=mini_batch_size)
            obs_t   = torch.tensor(real_buffer.obs[idx])
            act_t   = torch.tensor(real_buffer.act[idx])
            dhat_t  = torch.tensor(real_buffer.dhat[idx])   # (batch, DOB_DIM)

            inp    = torch.cat([obs_t, act_t], dim=-1)
            dx_res = res_net(inp)                            # (batch, DOB_DIM)
            loss   = nn.functional.mse_loss(dx_res, dhat_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum           += loss.item()
            loss_ct            += 1
            sampled_uncert_sum += float(uncert_mag[idx].mean())
            sampled_uncert_ct  += 1

    loss_avg         = loss_sum / max(1, loss_ct)
    sampled_uncert_avg = sampled_uncert_sum / max(1, sampled_uncert_ct)
    return loss_avg, sampled_uncert_avg


def train_uncertainty_rbf(uncert_model, optimizer, real_buffer,
                           res_net, batch_size: int, epochs: int) -> float:
    """
    MATLAB: trainUncertaintyRBF
    fresh_uncertainty_all = |Fpinv*e - dxRes| 를 한 번 계산 후 반복 학습.
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
        dx_res_all = res_net(dl_in_all).cpu().numpy()    # (N, DOB_DIM)

    dx_real_all  = next_obs_all - obs_all              # (N, OBS_DIM)
    e_all        = dx_real_all - dx_nom_all            # (N, OBS_DIM)
    fpinv_e      = e_all @ FPINV.T                     # (N, DOB_DIM)
    fresh_uncert = fpinv_e - dx_res_all                # (N, DOB_DIM)
    fresh_uncert_avg = float(np.linalg.norm(fresh_uncert, axis=1).mean())

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

            pred = uncert_model(inp_t)                   # (batch, DOB_DIM)
            loss = nn.functional.mse_loss(pred, target_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            ct       += 1

    return loss_sum / max(1, ct), fresh_uncert_avg


def evaluate_rbf_calibration(uncert_model, real_buffer,
                              sample_size: int = 4096):
    """
    real_buffer 샘플에서 RBF 예측 uncertainty와 실제 uncertainty를 비교한다.

    Returns
    -------
    calib_ratio : mean(||RBF pred||₂) / mean(||actual||₂)
    calib_corr  : Pearson correlation(pred_mag, actual_mag)
    """
    valid_len = real_buffer.length
    if valid_len == 0:
        return float('nan'), float('nan')

    n   = min(valid_len, sample_size)
    idx = np.random.choice(valid_len, size=n, replace=False)

    obs           = real_buffer.obs[idx]
    act           = real_buffer.act[idx]
    actual_uncert = real_buffer.uncertainty[idx]   # (n, DOB_DIM)

    uncert_model.eval()
    with torch.no_grad():
        inp  = torch.tensor(np.concatenate([obs, act], axis=-1))
        pred = uncert_model(inp).cpu().numpy()     # (n, DOB_DIM)

    pred_mag   = np.linalg.norm(pred,           axis=1)   # (n,)
    actual_mag = np.linalg.norm(actual_uncert,  axis=1)   # (n,)

    calib_ratio = float(pred_mag.mean() / (actual_mag.mean() + 1e-8))

    if pred_mag.std() < 1e-8 or actual_mag.std() < 1e-8:
        calib_corr = float('nan')
    else:
        calib_corr = float(np.corrcoef(pred_mag, actual_mag)[0, 1])

    return calib_ratio, calib_corr
