"""Compatibility CLI/import wrapper; implementation is in wheel_legged_gym.evaluation.transitions."""
import sys
from pathlib import Path
_cli_package_root = str(Path(__file__).resolve().parents[1] / "plane")
if _cli_package_root not in sys.path:
    sys.path.insert(0, _cli_package_root)
from wheel_legged_gym.evaluation.transitions import response_metrics, summarize

import argparse
import json
from pathlib import Path
import hashlib
if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.write_text(json.dumps(summarize(json.loads(args.input.read_text())),indent=2)+'\n')
