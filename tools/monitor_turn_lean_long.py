"""CLI for the bounded turn-lean training monitor."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'plane'))

from wheel_legged_gym.app.turn_lean_monitor import main


if __name__ == '__main__':
    main(root)
