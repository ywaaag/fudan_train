"""CLI entry for a read-only training scalar summary."""
import sys
from pathlib import Path

package_root = str(Path(__file__).resolve().parents[1] / 'plane')
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from wheel_legged_gym.app.training_summary import main


if __name__ == '__main__':
    main()
