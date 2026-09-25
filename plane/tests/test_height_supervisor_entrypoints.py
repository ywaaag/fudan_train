"""Relocation preserves each supervisor body and removes simulator import coupling."""
import ast
import importlib
import json
from pathlib import Path
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
BODIES = json.loads((Path(__file__).parent/'fixtures/height_supervisor_bodies.json').read_text())


@pytest.mark.parametrize('name', sorted(BODIES))
def test_body_is_unchanged(name):
    tree = ast.parse((ROOT/'plane/wheel_legged_gym/app'/f'{name}.py').read_text())
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    count = 2 if name=='run_height_course' else 1
    actual = ast.Module(body=main.body[count:],type_ignores=[])
    expected = ast.parse(textwrap.dedent(BODIES[name]))
    assert ast.dump(actual,include_attributes=False)==ast.dump(expected,include_attributes=False)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert all(alias.name!='isaacgym' for alias in node.names)


@pytest.mark.parametrize('name', sorted(BODIES))
def test_import_does_not_start_or_read_job(name, monkeypatch):
    import argparse
    import subprocess
    def forbidden(*args,**kwargs):
        pytest.fail('Import attempted task I/O')
    monkeypatch.setattr(argparse.ArgumentParser,'parse_args',forbidden)
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(Path,'mkdir',forbidden)
    monkeypatch.setattr(Path,'read_text',forbidden)
    monkeypatch.setattr(Path,'write_text',forbidden)
    module=importlib.import_module('wheel_legged_gym.app.'+name)
    importlib.reload(module)
    assert callable(module.main)
