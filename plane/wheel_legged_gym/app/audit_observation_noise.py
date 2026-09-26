"""Run the explicit observation-noise diagnostic as an application boundary."""
import subprocess
import sys
from pathlib import Path


def main(root, argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in ('sampled', 'sampled_noisy'):
        raise ValueError('Expected sampled or sampled_noisy')
    entry = Path(root) / 'plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py'
    return subprocess.run(
        [sys.executable, str(entry), '--diagnostic-mode', arguments[0], *arguments[1:]],
    ).returncode
