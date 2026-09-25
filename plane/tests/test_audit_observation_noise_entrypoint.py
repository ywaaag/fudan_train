"""Noise diagnostic keeps its explicit mode and subprocess argument boundary."""
import ast
import subprocess
from pathlib import Path

import pytest

from wheel_legged_gym.app.audit_observation_noise import main


@pytest.mark.parametrize('mode', ['sampled', 'sampled_noisy'])
def test_noise_mode_and_literal_arguments(tmp_path, monkeypatch, mode):
    calls=[]
    monkeypatch.setattr(subprocess, 'run', lambda command, cwd: calls.append((command,cwd)) or type('R',(),{'returncode':7})())
    assert main(tmp_path, [mode, 'literal$(unchanged)', '--seed', '19']) == 7
    assert calls == [([
        __import__('sys').executable,
        str(tmp_path/'plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py'),
        '--diagnostic-mode', mode, 'literal$(unchanged)', '--seed', '19',
    ], tmp_path)]


def test_invalid_mode_is_rejected_before_process():
    with pytest.raises(ValueError, match='Expected sampled'):
        main(Path('/unused'), ['unexpected'])


def test_cli_has_no_process_call_at_module_level():
    tree=ast.parse((Path(__file__).resolve().parents[2]/'tools/audit_observation_noise.py').read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess'
                   for n in ast.walk(tree))
