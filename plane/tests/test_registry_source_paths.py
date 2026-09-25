"""Source snapshots must preserve actual config implementations after facade removal."""
import ast
from pathlib import Path
from types import SimpleNamespace
from shutil import copyfile
import ntpath
import os

from wheel_legged_gym.adapters.artifacts.source_snapshot import save_source_snapshot


def test_registry_archives_canonical_configs(tmp_path):
    plane=Path(__file__).parents[1]
    package=plane/'wheel_legged_gym'
    tree=ast.parse((package/'app/task_registry.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='TaskRegistry')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='save_cfgs')
    method.returns=None
    scope=dict(os=os,ntpath=ntpath,copyfile=copyfile,save_source_snapshot=save_source_snapshot,
               WHEEL_LEGGED_GYM_ROOT_DIR=str(plane),WHEEL_LEGGED_GYM_ENVS_DIR=str(package/'envs'))
    exec(compile(ast.Module(body=[method],type_ignores=[]),'registry_snapshot','exec'),scope)
    run=tmp_path/'run'
    scope['save_cfgs'](SimpleNamespace(log_dir=str(run)),'wheel_legged')
    for name in ['legged_robot_config.py','wheel_legged_config.py']:
        assert (run/name).read_bytes()==(package/'contracts'/name).read_bytes()
    assert (run/'terrain_generation.py').read_bytes()==(package/'adapters/isaacgym/terrain_generation.py').read_bytes()
