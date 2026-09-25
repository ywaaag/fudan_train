"""Compare workflow commands, written JSON and failure behavior with frozen app code."""
import ast
import copy
import json
from pathlib import Path
import textwrap

import pytest

from wheel_legged_gym.adapters.artifacts.job_files import JobFiles
from wheel_legged_gym.workflows import height_evaluation


def frozen_block():
    source=json.loads((Path(__file__).parent/'fixtures/height_supervisor_bodies.json').read_text())
    tree=ast.parse(textwrap.dedent(source['run_height_course']))
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For)
              and isinstance(n.target,ast.Name) and n.target.id=='stage')
    start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='records')
    end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr)
             and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
             and n.value.func.attr=='append')
    return compile(ast.Module(body=loop.body[start:end],type_ignores=[]),'old_height_evaluation','exec')


@pytest.mark.parametrize('mode',['pass','failed_row','empty','process_error','malformed'])
def test_workflow_matches_old_block(tmp_path, monkeypatch, mode):
    bank=[(-1.,0.,.35),(0.,1.,.45)]
    calls=[]
    def assess(row,stage):
        row['gate']={'passed':row['ok'],'failed_checks':[] if row['ok'] else ['height_tracking']}
        return row['gate']
    monkeypatch.setattr(height_evaluation,'assess_height_row',assess)
    def run(arguments,tag):
        calls.append((arguments,tag))
        if mode=='process_error' and len(calls)==2: raise RuntimeError('child failed')
        path=Path(next(x.split('=',1)[1] for x in arguments if isinstance(x,str) and x.startswith('--out=')))
        rows=[] if mode=='empty' else [{'ok':mode!='failed_row','value':tag}]
        path.write_text('broken' if mode=='malformed' else json.dumps({'results':rows}))
    scope=dict(candidate=Path('/checkpoint/model_7400.pt'),stage='dual',bank=bank,
               PLANE=Path('/plane'),job=tmp_path,run=run,json=json,assess_height_row=assess)
    error=None
    try: exec(frozen_block(),scope)
    except (RuntimeError,json.JSONDecodeError) as exc: error=exc
    expected_calls=copy.deepcopy(calls)
    output=tmp_path/'dual_acceptance.json'
    expected_bytes=output.read_bytes() if output.exists() else None
    if output.exists(): output.unlink()
    calls.clear()
    kwargs=dict(evaluation_script=Path('/plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py'),
                files=JobFiles(tmp_path),run=run)
    if error:
        with pytest.raises(type(error)):
            height_evaluation.evaluate_height_checkpoint(scope['candidate'],'dual',bank,**kwargs)
        assert not output.exists()
    else:
        result=height_evaluation.evaluate_height_checkpoint(scope['candidate'],'dual',bank,**kwargs)
        assert result==scope['result'] and output.read_bytes()==expected_bytes
        assert len(calls)==3
    assert calls==expected_calls
