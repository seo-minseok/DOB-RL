"""
main.py — 단일 시드 학습 진입점
Usage:
  python main.py --checkpoint-dir ./checkpoints --seed 1
  python main.py --checkpoint-dir ./checkpoints --seed 1 --resume
  python main.py --checkpoint-dir ./checkpoints --seed 1 --gpu 0
  python main.py --checkpoint-dir ./checkpoints --seed 1 --real-ratio 0.5
  python main.py --checkpoint-dir ./checkpoints --seed 1 --uncertainty-threshold 0.3
"""
import argparse
import dataclasses
import json
import os
import pickle

import torch

from dob_mbrl.training import train_DOB_core, DOBMBRLConfig


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='DOB-MBRL single-seed training')
    parser.add_argument('--checkpoint-dir', type=str,
                        default=os.path.join(_SCRIPT_DIR, 'checkpoints'),
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=1,
                        help='랜덤 시드 (1-based)')
    parser.add_argument('--resume', action='store_true',
                        help='마지막 체크포인트에서 재개')
    parser.add_argument('--real-ratio', type=float, default=None,
                        help='병렬 실험용 real_ratio 덮어쓰기 (미입력 시 config.py 값 사용)')
    parser.add_argument('--uncertainty-threshold', type=float, default=None,
                        help='실험용 uncertainty_threshold 덮어쓰기 (미입력 시 config.py 값 사용)')
    parser.add_argument('--num-threads', type=int, default=None,
                        help='PyTorch intra-op 스레드 수 (병렬 실행 시 CPU 분배용, 예: --num-threads 8)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='사용할 GPU 인덱스 (예: --gpu 0, --gpu 1). -1이면 CPU 사용')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(args.num_threads)
        print(f'[main] num_threads={args.num_threads}')

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    cfg = DOBMBRLConfig()
    if args.real_ratio is not None:
        cfg.real_ratio = args.real_ratio
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.real_ratio is not None:
        run_name = f'real_ratio={cfg.real_ratio}'
    else:
        run_name = f'real_ratio={cfg.real_ratio}'

    _here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_here, 'results', run_name)
    os.makedirs(results_dir, exist_ok=True)

    config_path = os.path.join(results_dir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)
        print(f'[main] config saved → {config_path}')

    print(f'[main] seed={args.seed}  real_ratio={cfg.real_ratio}  '
          f'rbf_width={cfg.rbf_width}  num_rbf_centers={cfg.num_rbf_centers}  '
          f'uncertainty_threshold={cfg.uncertainty_threshold}  '
          f'checkpoint_dir={args.checkpoint_dir}  resume={args.resume}  device={device}')
    print(f'[main] results_dir={results_dir}')

    rewards, steps, metrics = train_DOB_core(
        run_idx        = args.seed,
        result_queue   = None,
        checkpoint_dir = args.checkpoint_dir,
        resume         = args.resume,
        cfg            = cfg,
        results_dir    = results_dir,
        device         = device,
    )

    log_path = os.path.join(results_dir, f'seed_{args.seed}_result.pkl')
    with open(log_path, 'wb') as f:
        pickle.dump({'rewards': rewards, 'steps': steps, 'metrics': metrics}, f)
    print(f'[main] Saved log → {log_path}')


if __name__ == '__main__':
    main()
