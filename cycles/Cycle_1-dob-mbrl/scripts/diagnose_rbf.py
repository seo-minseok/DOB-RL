"""
diagnose_rbf.py — 학습 전/후 RBF 파라미터 사전 진단

사용법:
    python scripts/diagnose_rbf.py                         # 랜덤 롤아웃 (기본)
    python scripts/diagnose_rbf.py --K 1200 --w 0.5       # 파라미터 직접 지정
    python scripts/diagnose_rbf.py --compare               # 여러 설정 비교
    python scripts/diagnose_rbf.py --ckpt checkpoints/real_ratio=0.05/Resume_Seed2_Latest.pt
    python scripts/diagnose_rbf.py --ckpt_dir checkpoints/real_ratio=0.05  # 전체 시드 합산

진단 항목:
  1. nearest-center 거리 분포  — center가 실제 state 근처에 있는지
  2. phi_max 분포              — kernel 응답이 의미있는 수준인지
  3. 유효 랭크                 — center가 실제로 몇 개 방향을 구별하는지
  4. PASS / WARN / FAIL 판정
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

import torch
from dob_mbrl.models import NormalizedRBFModel
from dob_mbrl.models.actor_network import ActorNetwork
from dob_mbrl.envs.hopper_utils import make_hopper_env, reset_env, step_env
from dob_mbrl.dynamics.constants import PHYS_MIN, PHYS_MAX
from dob_mbrl.training.config import DOBMBRLConfig


# ── 판정 기준 ──────────────────────────────────────────────────────────────
THRESHOLDS = {
    'phi_max_p5_warn':  0.005,   # phi_max 하위 5%가 이 미만이면 WARN
    'phi_max_p5_fail':  0.001,   # phi_max 하위 5%가 이 미만이면 FAIL
    'uncovered_warn':   10.0,    # phi_max < 0.01인 비율(%) WARN 기준
    'uncovered_fail':   25.0,    # phi_max < 0.01인 비율(%) FAIL 기준
    'eff_rank_warn':    5.0,     # 유효 랭크 WARN 기준
}


def load_from_checkpoint(path: str) -> np.ndarray:
    """체크포인트의 real_buffer에서 obs-act 샘플 추출."""
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    rb   = ckpt['real_buffer']
    n    = rb['length']
    obs  = rb['obs'][:n].astype(np.float32)
    act  = rb['act'][:n].astype(np.float32)
    return np.concatenate([obs, act], axis=1)


def load_from_checkpoint_dir(ckpt_dir: str) -> np.ndarray:
    """디렉토리 내 모든 체크포인트 real_buffer를 합산해 반환."""
    import glob
    paths = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
    if not paths:
        raise FileNotFoundError(f'No .pt files found in {ckpt_dir}')
    chunks = []
    for p in paths:
        try:
            chunks.append(load_from_checkpoint(p))
            print(f'  로드: {os.path.basename(p)} ({len(chunks[-1]):,} samples)')
        except Exception as e:
            print(f'  건너뜀: {os.path.basename(p)} — {e}')
    return np.concatenate(chunks, axis=0)


def collect_policy_episode(ckpt_path: str, seed: int = 0) -> np.ndarray:
    """
    체크포인트의 actor로 한 에피소드 실행 후 obs-act 수집.
    actor.forward() (deterministic) 사용.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    actor = ActorNetwork()
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_hopper_env()
    obs = np.array(reset_env(env), dtype=np.float32)
    samples = []
    with torch.no_grad():
        while True:
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            act_t  = actor(obs_t)
            act    = act_t.squeeze(0).numpy()
            next_obs_raw, _, terminated, truncated, _ = step_env(env, act)
            samples.append(np.concatenate([obs, act]))
            obs = np.array(next_obs_raw, dtype=np.float32)
            if terminated or truncated:
                break
    env.close()
    return np.stack(samples).astype(np.float32)


def collect_random_samples(n_steps: int = 2000, seed: int = 0) -> np.ndarray:
    """랜덤 정책으로 Hopper state-action 샘플 수집."""
    np.random.seed(seed)
    env = make_hopper_env()
    obs = np.array(reset_env(env), dtype=np.float32)
    samples = []
    for _ in range(n_steps):
        act = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
        next_obs_raw, _, terminated, truncated, _ = step_env(env, act)
        samples.append(np.concatenate([obs, act]))
        obs = np.array(next_obs_raw, dtype=np.float32)
        if terminated or truncated:
            obs = np.array(reset_env(env), dtype=np.float32)
    env.close()
    return np.stack(samples).astype(np.float32)


def diagnose(K: int, w: float, n_steps: int = 2000, seed: int = 42,
             obs_act: np.ndarray = None) -> dict:
    """
    RBF 파라미터 (K, w)에 대한 사전 진단.

    Parameters
    ----------
    K        : center 수
    w        : RBF width
    n_steps  : 랜덤 롤아웃 수집 스텝 수 (obs_act 없을 때)
    seed     : 재현성 시드
    obs_act  : 미리 수집된 샘플 (None이면 랜덤 롤아웃 수행)

    Returns
    -------
    dict of metrics + 'verdict' key
    """
    if obs_act is None:
        print(f'  랜덤 롤아웃 {n_steps} steps 수집 중...')
        obs_act = collect_random_samples(n_steps, seed)

    pm = np.array(PHYS_MIN); px = np.array(PHYS_MAX)
    x_norm = torch.tensor(
        2.0 * (obs_act - pm) / (px - pm) - 1.0, dtype=torch.float32
    )

    torch.manual_seed(seed)
    model = NormalizedRBFModel(K, w)
    c = model.centers  # (14, K)

    with torch.no_grad():
        c_sq    = (c ** 2).sum(0)
        x_sq    = (x_norm ** 2).sum(1)
        d2      = c_sq.unsqueeze(0) + x_sq.unsqueeze(1) - 2.0 * (x_norm @ c)
        d2      = d2.clamp(min=0)
        d_min   = d2.min(1).values.sqrt()
        phi_all = torch.exp(-d2 / (2.0 * w ** 2))
        phi_max = phi_all.max(1).values
        phi_sum = phi_all.sum(1)
        phi_n   = phi_all / (phi_sum.unsqueeze(1) + 1e-8)

    # 유효 랭크 (서브샘플 최대 500개)
    n_svd = min(len(x_norm), 500)
    idx   = torch.randperm(len(x_norm))[:n_svd]
    _, S, _ = torch.linalg.svd(phi_n[idx], full_matrices=False)
    S_sq    = S ** 2
    eff_rank = (S_sq.sum() ** 2 / (S_sq ** 2).sum()).item()

    d_np   = d_min.numpy()
    phi_np = phi_max.numpy()
    ps_np  = phi_sum.numpy()

    uncovered_pct = float((phi_np < 0.01).mean() * 100)

    metrics = {
        'K': K, 'w': w, 'n_samples': len(obs_act),
        'd_p5':   float(np.percentile(d_np, 5)),
        'd_med':  float(np.median(d_np)),
        'd_p95':  float(np.percentile(d_np, 95)),
        'phi_max_p5':   float(np.percentile(phi_np, 5)),
        'phi_max_med':  float(np.median(phi_np)),
        'phi_max_p95':  float(np.percentile(phi_np, 95)),
        'phi_sum_min':  float(ps_np.min()),
        'uncovered_pct': uncovered_pct,
        'eff_rank': float(eff_rank),
    }

    # 판정
    t = THRESHOLDS
    if (metrics['phi_max_p5'] < t['phi_max_p5_fail'] or
            metrics['uncovered_pct'] > t['uncovered_fail']):
        verdict = 'FAIL'
    elif (metrics['phi_max_p5'] < t['phi_max_p5_warn'] or
          metrics['uncovered_pct'] > t['uncovered_warn'] or
          metrics['eff_rank'] < t['eff_rank_warn']):
        verdict = 'WARN'
    else:
        verdict = 'PASS'

    metrics['verdict'] = verdict
    return metrics


def print_report(m: dict) -> None:
    v = m['verdict']
    badge = {'PASS': '✓ PASS', 'WARN': '△ WARN', 'FAIL': '✗ FAIL'}[v]
    print()
    print(f'┌─ RBF 사전 진단  K={m["K"]}  w={m["w"]}  [{badge}] ────────────')
    print(f'│  샘플 수: {m["n_samples"]:,}')
    print(f'│')
    print(f'│  [nearest-center 거리]')
    print(f'│    p5={m["d_p5"]:.3f}  median={m["d_med"]:.3f}  p95={m["d_p95"]:.3f}')
    print(f'│    → 클수록 center가 희박. w={m["w"]} 기준 median phi_nearest='
          f'{np.exp(-m["d_med"]**2/(2*m["w"]**2)):.4f}')
    print(f'│')
    print(f'│  [phi_max 분포]')
    print(f'│    p5={m["phi_max_p5"]:.4f}  median={m["phi_max_med"]:.4f}  p95={m["phi_max_p95"]:.4f}')
    print(f'│    phi_max < 0.01 (커버 밖): {m["uncovered_pct"]:.1f}%'
          + ('  ← WARN' if m["uncovered_pct"] > THRESHOLDS['uncovered_warn'] else ''))
    print(f'│')
    print(f'│  [표현력]')
    print(f'│    유효 랭크: {m["eff_rank"]:.1f}  (1=상수 출력, 높을수록 구별력 ↑)'
          + ('  ← WARN' if m["eff_rank"] < THRESHOLDS['eff_rank_warn'] else ''))
    print(f'│    phi_sum 최솟값: {m["phi_sum_min"]:.2e}  '
          f'(1e-8 안정화 대비 {m["phi_sum_min"]/1e-8:.0f}x)')
    print(f'└──────────────────────────────────────────────────────')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K',        type=int,   default=None)
    parser.add_argument('--w',        type=float, default=None)
    parser.add_argument('--n_steps',  type=int,   default=2000)
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--compare',  action='store_true',
                        help='config 기본값과 대안 설정을 같이 비교')
    parser.add_argument('--ckpt',     type=str,   default=None,
                        help='체크포인트 파일 경로 (.pt) — real_buffer 또는 episode 롤아웃')
    parser.add_argument('--ckpt_dir', type=str,   default=None,
                        help='체크포인트 디렉토리 — 전체 .pt 합산')
    parser.add_argument('--episode',  action='store_true',
                        help='--ckpt의 actor로 한 에피소드 실행 (real_buffer 대신)')
    args = parser.parse_args()

    cfg = DOBMBRLConfig()
    K = args.K if args.K is not None else cfg.num_rbf_centers
    w = args.w if args.w is not None else cfg.rbf_width

    # 샘플 수집
    if args.ckpt and args.episode:
        print(f'학습된 정책으로 에피소드 실행: {args.ckpt}')
        obs_act = collect_policy_episode(args.ckpt, seed=args.seed)
        print(f'에피소드 길이: {len(obs_act)} steps')
    elif args.ckpt_dir:
        print(f'체크포인트 디렉토리 로드: {args.ckpt_dir}')
        obs_act = load_from_checkpoint_dir(args.ckpt_dir)
        print(f'총 {len(obs_act):,} samples 합산')
    elif args.ckpt:
        print(f'체크포인트 로드: {args.ckpt}')
        obs_act = load_from_checkpoint(args.ckpt)
        print(f'총 {len(obs_act):,} samples')
    else:
        print(f'랜덤 롤아웃 샘플 수집 ({args.n_steps} steps)...')
        obs_act = collect_random_samples(args.n_steps, args.seed)

    if args.compare:
        for k_test, w_test in [(K, w), (1200, 0.3), (3000, 0.5)]:
            m = diagnose(k_test, w_test, obs_act=obs_act.copy(), seed=args.seed)
            print_report(m)
    else:
        m = diagnose(K, w, obs_act=obs_act, seed=args.seed)
        print_report(m)

        if m['verdict'] == 'FAIL':
            print()
            print('권장 조치: width를 높이거나 (예: 0.5→0.7) K를 늘리세요.')
        elif m['verdict'] == 'WARN':
            print()
            print('주의: 현재 설정으로 학습 가능하지만 표현력이 제한될 수 있습니다.')
