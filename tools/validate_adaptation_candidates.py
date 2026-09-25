"""CLI entry; task orchestration lives in the application module."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
package_root = str(ROOT / 'plane')
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from wheel_legged_gym.app.validate_adaptation_candidates import main


if __name__ == '__main__':
    main(ROOT)
