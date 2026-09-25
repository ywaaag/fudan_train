"""Compatibility CLI for the public MuJoCo tree ramp probe."""
import os
import sys
from pathlib import Path


def main():
    training_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(training_root / 'plane'))
    from wheel_legged_gym.adapters.mujoco.api import run_tree_probe
    root = os.environ.get('FUDAN_SIM2SIM_ROOT', str(training_root.parent / 'wheel_leg_sim2sim'))
    return run_tree_probe(sys.argv[1:] + ['--training-root', str(training_root)],
                          repository=root, interpreter=sys.executable)


if __name__ == '__main__':
    raise SystemExit(main())
