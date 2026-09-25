"""Compatibility CLI/import wrapper; implementation is in wheel_legged_gym.domain.commands.ramp."""
import sys
from pathlib import Path
_cli_package_root = str(Path(__file__).resolve().parents[1] / "plane")
if _cli_package_root not in sys.path:
    sys.path.insert(0, _cli_package_root)
from wheel_legged_gym.domain.commands.ramp import ramp_fraction
