"""Compatibility CLI for the public MuJoCo validation process."""
import os
import sys
from pathlib import Path


def main():
    # Bootstrap only; business modules never mutate sys.path.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"plane"))
    from wheel_legged_gym.adapters.mujoco.api import run_validation
    root=os.environ.get("FUDAN_SIM2SIM_ROOT",str(Path(__file__).resolve().parents[2]/"wheel_leg_sim2sim"))
    return run_validation(sys.argv[1:],repository=root,interpreter=sys.executable)


if __name__=="__main__":raise SystemExit(main())
