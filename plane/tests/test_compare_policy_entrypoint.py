"""Historical policy comparison now has an explicit application entry."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

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


def test_main_uses_explicit_root_and_records_app_source_hash(tmp_path, monkeypatch):
    module=importlib.import_module('wheel_legged_gym.app.compare_policy_versions')
    plane=tmp_path/'plane'
    source=plane/'logs/wheel_legged/sample'
    source.mkdir(parents=True)
    (source/'model_1.pt').write_bytes(b'checkpoint')
    (source/'policy_experiment.json').write_text('{}')
    evaluator=plane/'wheel_legged_gym/scripts/evaluate_policy_comparison.py'
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text('# test evaluator\n')
    assets=tmp_path/'assets'
    assets.mkdir()
    (assets/'wheel_leg_train.urdf').write_text('<robot/>')
    (plane/'outputs').mkdir()
    monkeypatch.setattr(module,'CANDIDATES',{'sample':('sample',1)})
    monkeypatch.setattr(module.subprocess,'check_output',lambda command,**kwargs: 'test-git')
    launched=[]
    def fake_popen(command, **kwargs):
        launched.append((command,kwargs['cwd']))
        raise RuntimeError('fake child stopped')
    monkeypatch.setattr(module.subprocess,'Popen',fake_popen)
    monkeypatch.setattr(sys,'argv',['compare_policy_versions.py','--candidates','sample',
                                    '--seeds','19','--randomization-levels','0'])
    with pytest.raises(RuntimeError,match='fake child stopped'):
        module.main(tmp_path)
    jobs=list((plane/'outputs').glob('policy_comparison_*'))
    assert len(jobs)==1
    manifest=json.loads((jobs[0]/'manifest.json').read_text())
    assert manifest['candidates']['sample']['checkpoint']==str(source/'model_1.pt')
    assert manifest['runner_sha256']==module.sha(module.__file__)
    assert manifest['evaluator_sha256']==module.sha(evaluator)
    assert launched[0][0][1]==str(evaluator)
    assert launched[0][1]==tmp_path


def test_pre_migration_runner_hash_rejects_resume(tmp_path, monkeypatch):
    module=importlib.import_module('wheel_legged_gym.app.compare_policy_versions')
    evaluator=tmp_path/'plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py'
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text('# test evaluator\n')
    job=tmp_path/'old_job'
    job.mkdir()
    (job/'manifest.json').write_text(json.dumps({
        'evaluator_sha256':module.sha(evaluator),
        'runner_sha256':module.sha(ROOT/'tools/compare_policy_versions.py'),
    }))
    monkeypatch.setattr(sys,'argv',['compare_policy_versions.py','--job',str(job)])
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k: (_ for _ in ()).throw(AssertionError('launched')))
    with pytest.raises(RuntimeError,match='Evaluation code changed; use a new job'):
        module.main(tmp_path)
