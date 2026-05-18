"""
diagnose_is_done.py
--------------------
DOB-RL 모델과 MBRL 모델이 is_done을 얼마나 정확하게 예측하는지 비교.

핵심 질문:
  real_buffer에 done=True 전이가 있을 때,
  각 모델이 예측한 next_obs로 reward_is_done_function을 계산하면
  is_done=True를 제대로 예측하는가?

사용법:
  cd cycles/Cycle_5
  python diagnose_is_done.py --dob-ckpt checkpoints/real_ratio=0.2_uncert_thresh=0.4/Champion_Seed1_BestModel.pt
                             --mbrl-ckpt checkpoints/mbrl_real_ratio=0.2/MBRL_Seed1_BestModel.pt
"""
import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dob_mbrl.envs.bipedalwalker_utils import (
    make_bipedalwalker_env, reset_env, step_env, reward_is_done_function,
)
from dob_mbrl.dynamics import (
    default_bipedalwalker_params, step_nominal_bipedalwalker, FPINV, F_MAT,
)
from dob_mbrl.dynamics.dob import predict_next_obs_dob
from dob_mbrl.models import ResidualDxNet, ContactNet, TransitionNetwork
from dob_mbrl.dynamics.constants import OBS_DIM, ACT_DIM


def collect_transitions(num_episodes: int = 100):
    """
    실제 환경에서 무작위 policy로 전이 수집.
    terminal 전이(done=True 직전 스텝)를 포함.
    Returns list of (obs, action, next_obs, done_real).
    """
    env   = make_bipedalwalker_env()
    trans = []
    for _ in range(num_episodes):
        obs = reset_env(env)
        for _ in range(1600):
            action   = np.random.uniform(-1, 1, size=ACT_DIM).astype(np.float32)
            next_obs, _, done, _ = step_env(env, action)
            trans.append((obs.copy(), action.copy(), next_obs.copy(), done))
            if done:
                break
            obs = next_obs
    env.close()
    return trans


def predict_done_dob(transitions, res_net, contact_net, p_nom):
    """DOB-RL 모델로 is_done 예측."""
    results = []
    for obs, act, next_obs_real, done_real in transitions:
        pred_next = predict_next_obs_dob(
            obs.reshape(1, -1), act.reshape(1, -1),
            res_net, p_nom, use_nominal=True, contact_net=contact_net,
        )
        _, is_done_model = reward_is_done_function(
            obs.reshape(1, -1), act.reshape(1, -1), pred_next
        )
        results.append({
            'done_real' : bool(done_real),
            'done_model': bool(is_done_model[0]),
        })
    return results


def predict_done_mbrl(transitions, transition_models):
    """MBRL 앙상블 모델로 is_done 예측 (과반수 투표)."""
    results = []
    for obs, act, next_obs_real, done_real in transitions:
        obs_t = torch.tensor(obs.reshape(1, -1))
        act_t = torch.tensor(act.reshape(1, -1))
        votes = []
        for tm in transition_models:
            tm.eval()
            with torch.no_grad():
                dx       = tm(obs_t, act_t).numpy()
            pred_next = obs.reshape(1, -1) + dx
            _, is_done_m = reward_is_done_function(
                obs.reshape(1, -1), act.reshape(1, -1), pred_next
            )
            votes.append(bool(is_done_m[0]))
        # 과반수 투표
        done_model = sum(votes) > len(votes) / 2
        results.append({
            'done_real' : bool(done_real),
            'done_model': done_model,
        })
    return results


def print_confusion(label: str, results: list):
    total     = len(results)
    real_pos  = [r for r in results if r['done_real']]
    real_neg  = [r for r in results if not r['done_real']]

    tp = sum(1 for r in real_pos if r['done_model'])     # 실제 done, 모델도 done
    fn = sum(1 for r in real_pos if not r['done_model']) # 실제 done, 모델은 not-done (False Negative)
    fp = sum(1 for r in real_neg if r['done_model'])     # 실제 not-done, 모델은 done (False Positive)
    tn = sum(1 for r in real_neg if not r['done_model']) # 실제 not-done, 모델도 not-done

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  전체 전이 수       : {total}")
    print(f"  실제 done=True 수  : {len(real_pos)} ({100*len(real_pos)/total:.1f}%)")
    print(f"  실제 done=False 수 : {len(real_neg)} ({100*len(real_neg)/total:.1f}%)")
    print()
    print(f"  [핵심] False Negative (실제 done=T, 모델 done=F): {fn}/{len(real_pos)} = {100*fn/max(1,len(real_pos)):.1f}%")
    print(f"         True  Positive (실제 done=T, 모델 done=T): {tp}/{len(real_pos)} = {100*tp/max(1,len(real_pos)):.1f}%")
    print(f"  [참고] False Positive (실제 done=F, 모델 done=T): {fp}/{len(real_neg)} = {100*fp/max(1,len(real_neg)):.1f}%")
    print(f"         True  Negative (실제 done=F, 모델 done=F): {tn}/{len(real_neg)} = {100*tn/max(1,len(real_neg)):.1f}%")
    print()

    if len(real_pos) > 0:
        recall    = tp / len(real_pos)
        precision = tp / max(1, tp + fp)
        print(f"  is_done Recall    (= TP rate): {recall:.3f}  ← 높을수록 쓰러짐 탐지 잘 함")
        print(f"  is_done Precision            : {precision:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dob-ckpt',  type=str,
                        default='checkpoints/real_ratio=0.2_uncert_thresh=0.4/Champion_Seed1_BestModel.pt')
    parser.add_argument('--mbrl-ckpt', type=str,
                        default='checkpoints/mbrl_real_ratio=0.2/MBRL_Seed1_BestModel.pt')
    parser.add_argument('--episodes',  type=int, default=50,
                        help='환경 episode 수 (많을수록 정확, 느림)')
    args = parser.parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))

    print(f"[1] 환경 전이 수집 ({args.episodes} episodes)...")
    transitions = collect_transitions(num_episodes=args.episodes)
    done_count  = sum(1 for _, _, _, d in transitions if d)
    print(f"    총 {len(transitions)} 전이, done=True: {done_count} ({100*done_count/len(transitions):.1f}%)")

    # ── DOB-RL ────────────────────────────────────────────────────────────────
    dob_path = os.path.join(_here, args.dob_ckpt)
    if os.path.exists(dob_path):
        print(f"\n[2] DOB-RL 체크포인트 로드: {dob_path}")
        ckpt    = torch.load(dob_path, weights_only=False)
        res_net = ResidualDxNet(OBS_DIM, ACT_DIM, hidden=64)
        res_net.load_state_dict(ckpt['res_net'])
        res_net.eval()

        contact_net = ContactNet(OBS_DIM, ACT_DIM, hidden=64)
        if 'contact_net' in ckpt:
            contact_net.load_state_dict(ckpt['contact_net'])
        contact_net.eval()

        p_nom   = default_bipedalwalker_params()
        results_dob = predict_done_dob(transitions, res_net, contact_net, p_nom)
        print_confusion("DOB-RL (nominal + ResidualNet + ContactNet)", results_dob)
    else:
        print(f"\n[!] DOB-RL checkpoint not found: {dob_path}")
        results_dob = None

    # ── MBRL ──────────────────────────────────────────────────────────────────
    mbrl_path = os.path.join(_here, args.mbrl_ckpt)
    if os.path.exists(mbrl_path):
        print(f"\n[3] MBRL 체크포인트 로드: {mbrl_path}")
        ckpt_mbrl = torch.load(mbrl_path, weights_only=False)
        num_tm    = 3
        tms = [TransitionNetwork(OBS_DIM, ACT_DIM, hidden=256) for _ in range(num_tm)]
        for i, tm in enumerate(tms):
            key = f'transition_model_{i}'
            if key in ckpt_mbrl:
                tm.load_state_dict(ckpt_mbrl[key])
            tm.eval()

        results_mbrl = predict_done_mbrl(transitions, tms)
        print_confusion("MBRL (3 × TransitionNet-256)", results_mbrl)
    else:
        print(f"\n[!] MBRL checkpoint not found: {mbrl_path}")
        results_mbrl = None

    # ── 가설 검증 요약 ─────────────────────────────────────────────────────────
    if results_dob is not None and results_mbrl is not None:
        real_pos_dob  = [r for r in results_dob  if r['done_real']]
        real_pos_mbrl = [r for r in results_mbrl if r['done_real']]

        fn_rate_dob  = sum(1 for r in real_pos_dob  if not r['done_model']) / max(1, len(real_pos_dob))
        fn_rate_mbrl = sum(1 for r in real_pos_mbrl if not r['done_model']) / max(1, len(real_pos_mbrl))

        print(f"\n{'='*55}")
        print(f"  가설 검증 요약")
        print(f"{'='*55}")
        print(f"  DOB-RL  False Negative rate: {fn_rate_dob:.3f}")
        print(f"  MBRL    False Negative rate: {fn_rate_mbrl:.3f}")
        print()
        if fn_rate_dob > fn_rate_mbrl + 0.1:
            print("  → 가설 지지: DOB-RL 모델이 쓰러짐을 훨씬 더 많이 놓침.")
            print("    model_buffer에 -100 reward가 MBRL보다 적게 들어가고,")
            print("    TD3 critic이 쓰러짐의 심각성을 학습하지 못함.")
        elif abs(fn_rate_dob - fn_rate_mbrl) <= 0.1:
            print("  → 가설 기각 혹은 약지지: is_done 예측 정확도가 비슷함.")
            print("    다른 원인(reward 크기 오차, Q-value 발산 등)을 추가 조사 필요.")
        else:
            print("  → 예상 외: MBRL 모델이 오히려 is_done을 더 많이 놓침.")


if __name__ == '__main__':
    main()
