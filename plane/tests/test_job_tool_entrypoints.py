"""Old job-file tools are thin CLI leaves with explicit app implementations."""
import ast
import importlib
import runpy
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


def test_registry_cli_passes_repository_root_without_export(monkeypatch):
    module=importlib.import_module('wheel_legged_gym.app.export_model_registry')
    roots=[]
    monkeypatch.setattr(module,'main',lambda root: roots.append(root))
    runpy.run_path(str(ROOT/'tools/export_model_registry.py'),run_name='__main__')
    assert roots==[ROOT]
