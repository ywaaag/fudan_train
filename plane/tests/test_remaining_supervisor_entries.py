"""Batch relocation preserves executable AST and leaves task I/O behind main."""
import ast
import importlib
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
SOURCES=json.loads((Path(__file__).parent/'fixtures/supervisor_entry_sources.json').read_text())


@pytest.mark.parametrize('name',sorted(SOURCES))
def test_entry_body_and_helpers_preserved(name):
    old=ast.parse(SOURCES[name])
    new=ast.parse((ROOT/'plane/wheel_legged_gym/app'/f'{name}.py').read_text())
    old_main=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    new_main=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    constants=[n for n in old.body if isinstance(n,ast.Assign)
               and n.targets[0].id not in ['ROOT','_cli_package_root']]
    if name in ['run_stand_ablation','train_stand_long']:
        old_main.body=old_main.body[1:]
    old_body=ast.Module(body=constants+old_main.body,type_ignores=[])
    new_body=ast.Module(body=new_main.body[1:],type_ignores=[])
    assert ast.dump(old_body,include_attributes=False)==ast.dump(new_body,include_attributes=False)
    old_helpers=[n for n in old.body if isinstance(n,ast.FunctionDef) and n.name!='main']
    new_helpers=[n for n in new.body if isinstance(n,ast.FunctionDef) and n.name!='main']
    assert ast.dump(ast.Module(body=old_helpers,type_ignores=[]))==ast.dump(ast.Module(body=new_helpers,type_ignores=[]))


@pytest.mark.parametrize('name',sorted(SOURCES))
def test_application_import_has_no_task_io(name,monkeypatch):
    import argparse
    import subprocess
    def forbidden(*args,**kwargs): pytest.fail('Import attempted task I/O')
    monkeypatch.setattr(argparse.ArgumentParser,'parse_args',forbidden)
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(Path,'mkdir',forbidden)
    monkeypatch.setattr(Path,'read_text',forbidden)
    monkeypatch.setattr(Path,'write_text',forbidden)
    module=importlib.import_module('wheel_legged_gym.app.'+name)
    importlib.reload(module)
    assert callable(module.main)
