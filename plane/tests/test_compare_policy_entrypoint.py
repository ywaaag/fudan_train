"""Historical policy comparison now has an explicit application entry."""
import ast
import importlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def test_cli_is_thin_and_application_import_is_side_effect_free(monkeypatch):
    cli=ast.parse((ROOT/'tools/compare_policy_versions.py').read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess'
                   for n in ast.walk(cli))
    import argparse, subprocess
    def forbidden(*args,**kwargs): raise AssertionError('comparison must not start on import')
    monkeypatch.setattr(argparse.ArgumentParser,'parse_args',forbidden)
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(Path,'mkdir',forbidden)
    monkeypatch.setattr(Path,'read_text',forbidden)
    module=importlib.import_module('wheel_legged_gym.app.compare_policy_versions')
    importlib.reload(module)
    assert callable(module.main)
