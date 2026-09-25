"""Legacy supervisors preserve their body but cannot execute during import."""
import ast
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BODIES = json.loads((Path(__file__).parent/'fixtures/legacy_supervisor_bodies.json').read_text())


@pytest.mark.parametrize('name', sorted(BODIES))
def test_moved_supervisor_body_matches_original(name):
    tree = ast.parse((ROOT/'plane/wheel_legged_gym/app'/f'{name}.py').read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='main')
    # Only the injected root normalization was added before the unchanged body.
    body = ast.Module(body=main.body[1:], type_ignores=[])
    expected = ast.parse(BODIES[name])
    if name == 'run_low_speed':
        for node in ast.walk(expected):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'all':
                node.value = ast.parse('low_speed_passed(reports)', mode='eval').body
    if name == 'stand_long_guard':
        # Gate extraction is separately compared against this frozen code.
        expected.body = [node for node in expected.body
                         if not (isinstance(node, ast.FunctionDef) and node.name == 'healthy')]
        loop = next(node for node in ast.walk(expected) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'segment')
        start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'reasons')
        end = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and node.value.func.attr == 'append')
        audits_assignment = loop.body[start + 1]
        loop.body[start:end] = [audits_assignment, ast.parse(
            'reasons,drift=assess_segment(audits,base_audits,current,base_stationary)').body[0]]
    assert ast.dump(body, include_attributes=False) == ast.dump(expected, include_attributes=False)


@pytest.mark.parametrize('name', sorted(BODIES))
def test_import_has_no_job_side_effects(name, monkeypatch):
    import argparse
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError('Import must not parse CLI, write files or launch a process')
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', forbidden)
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    monkeypatch.setattr(Path, 'read_text', forbidden)
    monkeypatch.setattr(Path, 'write_text', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    module = importlib.import_module('wheel_legged_gym.app.'+name)
    importlib.reload(module)
    assert callable(module.main)
