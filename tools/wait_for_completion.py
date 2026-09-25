"""CLI entry for wait_for_completion."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
package_root = str(ROOT / 'plane')
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from wheel_legged_gym.app.wait_for_completion import main


if __name__ == '__main__':
    main()
