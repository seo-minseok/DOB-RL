"""
main.py — 단일 시드 학습 진입점
Usage:
  python main.py --checkpoint-dir ./checkpoints --seed 1
  python main.py --checkpoint-dir ./checkpoints --seed 1 --resume
  python main.py --checkpoint-dir ./checkpoints --seed 1 --real-ratio 0.75
  python main.py --checkpoint-dir ./checkpoints --seed 1 --uncertainty-threshold 0.3
"""
import argparse
import dataclasses
import json
import os

from dob_mbrl.training import train_DOB_core, DOBMBRLConfig


def parse_args():
    parser = argparse.ArgumentParser(description='DOB-MBRL single-seed training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=1,
                        help='랜덤 시드 (1-based)')
    parser.add_argument('--resume', action='store_true',
                        help='마지막 체크포인트에서 재개')
    parser.add_argument('--real-ratio', type=float, default=None,
                        help='병렬 실험용 real_ratio 덮어쓰기 (미입력 시 config.py 값 사용)')
    parser.add_argument('--uncertainty-threshold', type=float, default=None,
                        help='실험용 uncertainty_threshold 덮어쓰기 (미입력 시 config.py 값 사용)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = DOBMBRLConfig()

    if args.real_ratio is not None:
        cfg.real_ratio = args.real_ratio
    if args.uncertainty_threshold is not None:
        cfg.uncertainty_threshold = args.uncertainty_threshold

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.uncertainty_threshold is not None:
        run_name = f'uncert_thresh={cfg.uncertainty_threshold}'
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

    print(f'[main] seed={args.seed}  real_ratio={cfg.real_ratio}  uncertainty_threshold={cfg.uncertainty_threshold}  checkpoint_dir={args.checkpoint_dir}  resume={args.resume}')
    print(f'[main] results_dir={results_dir}')

    rewards, steps = train_DOB_core(
        run_idx        = args.seed,
        num_episodes   = cfg.num_episodes,
        result_queue   = None,
        checkpoint_dir = args.checkpoint_dir,
        resume         = args.resume,
        cfg            = cfg,
        results_dir    = results_dir,
    )


if __name__ == '__main__':
    main()
