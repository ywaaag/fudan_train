"""Height continuation CLI does not start a round during import."""
import ast
import importlib
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def test_cli_has_no_process_side_effect_and_app_has_explicit_root(monkeypatch):
    tree=ast.parse((ROOT/'tools/continue_height_course.py').read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess'
                   for n in ast.walk(tree))
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(Path,'mkdir',lambda *a,**k: (_ for _ in ()).throw(AssertionError()))
    module=importlib.import_module('wheel_legged_gym.app.continue_height_course')
    assert next(n for n in ast.walk(ast.parse(Path(module.__file__).read_text()))
                if isinstance(n,ast.FunctionDef) and n.name=='main').args.args[0].arg=='root'
