"""Print compact TensorBoard tail statistics without dumping training stdout."""
import argparse
import json
from pathlib import Path

from wheel_legged_gym.adapters.artifacts.tensorboard_scalars import read_scalars
from wheel_legged_gym.evaluation.training_summary import METRIC_KEYS, summarize_scalars


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args(argv)
    samples = read_scalars(args.run, METRIC_KEYS)
    print(json.dumps(summarize_scalars(args.run, samples, args.window), indent=2))
