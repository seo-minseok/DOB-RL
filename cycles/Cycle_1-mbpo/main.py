"""
main.py — 단일 시드 학습 진입점
Usage:
  python main.py --checkpoint-dir ./checkpoints --seed 1
  python main.py --checkpoint-dir ./checkpoints --seed 1 --resume
  python main.py --checkpoint-dir ./checkpoints --seed 1 --gpu 0
"""
import argparse
import dataclasses
import json
import os
import pickle

import torch

from mbpo.training import train_MBPO_core, MBPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description='MBPO single-seed training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=1, help='랜덤 시드 (1-based)')
    parser.add_argument('--resume', action='store_true', help='마지막 체크포인트에서 재개')
    parser.add_argument('--num-threads', type=int, default=None,
                        help='PyTorch intra-op 스레드 수')
    parser.add_argument('--gpu', type=int, default=0,
                        help='사용할 GPU 인덱스 (-1이면 CPU)')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(args.num_threads)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    cfg = MBPOConfig()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    _here       = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_here, 'results', 'default')
    os.makedirs(results_dir, exist_ok=True)

    config_path = os.path.join(results_dir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)
        print(f'[main] config saved → {config_path}')

    print(f'[main] seed={args.seed}  total_steps={cfg.total_steps}  '
          f'real_ratio={cfg.real_ratio}  device={device}')

    rewards, steps, metrics = train_MBPO_core(
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
