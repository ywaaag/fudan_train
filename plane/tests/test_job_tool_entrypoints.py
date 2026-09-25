"""Old job-file tools are thin CLI leaves with explicit app implementations."""
import ast
import importlib
import subprocess
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name', ['wait_for_completion','summarize_policy_comparison','export_model_registry'])
def test_job_tool_imports_are_side_effect_free(name, monkeypatch):
    source=(ROOT/'tools'/f'{name}.py').read_text()
    tree=ast.parse(source)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess'
                   for n in ast.walk(tree))
    monkeypatch.setattr(Path,'read_text',lambda *a,**k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(Path,'write_text',lambda *a,**k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(Path,'mkdir',lambda *a,**k: (_ for _ in ()).throw(AssertionError()))
    module=importlib.import_module('wheel_legged_gym.app.'+name)
    importlib.reload(module)
    assert callable(module.main)
