import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

sys.path.insert(0, r'C:\Users\seominseok\DOB-RL\cycles\Cycle_8')
from dob_mbrl.training.trainer import train_DOB_core
from dob_mbrl.training.config import DOBMBRLConfig

cfg = DOBMBRLConfig(real_ratio=1.0)
train_DOB_core(
    args.seed,
    cfg.num_episodes,
    cfg=cfg,
    results_dir=r'C:\Users\seominseok\DOB-RL\cycles\Cycle_8\results\real_ratio=1.0',
    checkpoint_dir=r'C:\Users\seominseok\DOB-RL\cycles\Cycle_8\checkpoints',
)
