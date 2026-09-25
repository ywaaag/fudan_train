"""MuJoCo CLI boundary preserves arguments and exit status without importing MuJoCo."""
import json
import sys

import pytest

from wheel_legged_gym.adapters.mujoco.api import run_mapping_audit, run_validation, run_tree_probe


@pytest.mark.parametrize('run,name', [(run_mapping_audit, 'audit_mapping.py'),
                                      (run_validation, 'validate_policy.py'),
                                      (run_tree_probe, 'probe_tree_ramp.py')])
def test_process_boundary_passes_literal_arguments_and_exit_code(tmp_path, run, name):
    record = tmp_path / 'received.json'
    (tmp_path / name).write_text(
        'import sys,json\nfrom pathlib import Path\n'
        'Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n'
        'raise SystemExit(7)\n'
    )
    arguments = ['file with spaces.json', '--out', 'literal$(not-a-shell).json']
    assert run([str(record)] + arguments, repository=tmp_path, interpreter=sys.executable) == 7
    assert json.loads(record.read_text()) == arguments


def test_missing_public_audit_entry_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match='public mapping audit entry missing'):
        run_mapping_audit([], repository=tmp_path, interpreter=sys.executable)
