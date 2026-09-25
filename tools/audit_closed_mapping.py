"""Compatibility CLI for the MuJoCo repository's public mapping audit."""
import os
import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'plane'))
    from wheel_legged_gym.adapters.mujoco.api import run_mapping_audit
    root = os.environ.get('FUDAN_SIM2SIM_ROOT',
                          str(Path(__file__).resolve().parents[2] / 'wheel_leg_sim2sim'))
    return run_mapping_audit(sys.argv[1:], repository=root, interpreter=sys.executable)


if __name__ == '__main__':
    raise SystemExit(main())
