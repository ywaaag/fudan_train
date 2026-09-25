"""Dual-height gates match frozen comparisons, including missing response and NaN."""
import ast
import copy
import json
from pathlib import Path
import textwrap

import pytest

from wheel_legged_gym.evaluation import height_acceptance


@pytest.mark.parametrize('height', [.014, .015, .015001, float('nan')])
@pytest.mark.parametrize('settling', [None, 2., 3., 3.001, float('nan')])
@pytest.mark.parametrize('base_passed', [False,True])
def test_dual_gates_against_frozen_code(monkeypatch,height,settling,base_passed):
    bodies=json.loads((Path(__file__).parent/'fixtures/height_supervisor_bodies.json').read_text())
    tree=ast.parse(textwrap.dedent(bodies['run_dual_height']))
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
           and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'passed','settled'}]
    records=[{'metrics':{'height_mae':height},'nonwheel_contact_full_fraction':0}]
    responses=[{'rows':[{'response':{'height':{'settled_envs':4,'total_envs':4,'worst_settling_s':settling}}}]}]
    def gate(row): return {'passed':base_passed}
    monkeypatch.setattr(height_acceptance,'gate',gate)
    scope=dict(records=records,responses=responses,gate=gate)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'old_dual_gates','exec'),scope)
    before=json.dumps([records,responses])
    assert height_acceptance.assess_height_transitions(records,responses)==(scope['passed'],scope['settled'])
    assert json.dumps([records,responses])==before


def test_contact_unsettled_and_empty_semantics(monkeypatch):
    monkeypatch.setattr(height_acceptance,'gate',lambda row:{'passed':True})
    records=[{'metrics':{'height_mae':0},'nonwheel_contact_full_fraction':.001}]
    responses=[{'rows':[{'response':{'height':{'settled_envs':3,'total_envs':4,'worst_settling_s':1}}}]}]
    assert height_acceptance.assess_height_transitions(records,responses)==(False,False)
    assert height_acceptance.assess_height_transitions([],[])==(True,True)
