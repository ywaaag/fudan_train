"""Compatibility CLI/import wrapper; implementation is in wheel_legged_gym.evaluation.sequences."""
import sys
from pathlib import Path
_cli_package_root = str(Path(__file__).resolve().parents[1] / "plane")
if _cli_package_root not in sys.path:
    sys.path.insert(0, _cli_package_root)
from wheel_legged_gym.evaluation.sequences import summarize

import argparse
import json
from pathlib import Path
import hashlib
if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    data=json.loads(a.input.read_text());report=summarize(data)
    report.update(source=str(a.input.resolve()),source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),policy_sha256=data['policy_sha256'])
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
