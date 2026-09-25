"""Legacy CLI delegating explicit noise/action options; no monkey patch."""
import subprocess
import sys
from pathlib import Path


def main():
    mode=sys.argv[1]
    if mode not in ('sampled','sampled_noisy'):raise ValueError('Expected sampled or sampled_noisy')
    entry=Path(__file__).resolve().parents[1]/'plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py'
    return subprocess.run([sys.executable,str(entry),'--diagnostic-mode',mode,*sys.argv[2:]]).returncode


if __name__=='__main__':raise SystemExit(main())
