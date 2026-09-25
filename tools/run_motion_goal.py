"""Compatibility CLI for the motion supervisor application."""
import sys
from pathlib import Path

_cli_package_root = str(Path(__file__).resolve().parents[1] / "plane")
if _cli_package_root not in sys.path:
    sys.path.insert(0, _cli_package_root)

from wheel_legged_gym.app.motion_supervisor import main as run_application, digest
from wheel_legged_gym.evaluation.motion_candidates import assess, geometry_retained


def main():
    return run_application(Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
