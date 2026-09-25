"""Frozen legacy gate covers exact speed boundaries and invalid audit data."""
import ast
import copy
import json
from pathlib import Path

import pytest

from wheel_legged_gym.evaluation.low_speed_relay import low_speed_passed


@pytest.mark.parametrize('vx', [-.5, 0., .5])
@pytest.mark.parametrize('case', ['pass', 'speed_equal', 'speed_over', 'yaw_over', 'contact', 'height', 'failure', 'empty'])
def test_low_speed_gate_against_old_supervisor(vx, case):
    source = json.loads((Path(__file__).parent/'fixtures/legacy_supervisor_bodies.json').read_text())
    tree = ast.parse(source['run_low_speed'])
    expression = next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                      and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                      and n.value.func.id=='all')
    raw = {'failure_count':0,'timeout_count':0,'command_vx':vx,'tracking_vx_mae':0.,
           'metrics':{'nonwheel_contact_fraction':{'mean':0.},'left_contact':{'mean':.99},
                      'right_contact':{'mean':1.},'height_m':{'mean':.4},'yaw_rad_s':{'mean_abs':.10}}}
    if case.startswith('speed'):
        raw['tracking_vx_mae']=(.05 if vx==0 else .10)+(1e-9 if case=='speed_over' else 0)
    if case=='yaw_over': raw['metrics']['yaw_rad_s']['mean_abs']=.100001
    if case=='contact': raw['metrics']['nonwheel_contact_fraction']['mean']=1e-9
    if case=='height': raw['metrics']['height_m']['mean']=.44
    if case=='failure': raw['failure_count']=1
    reports=[] if case=='empty' else [raw]
    before=copy.deepcopy(reports)
    old=eval(compile(ast.Expression(expression),'frozen_low_speed_gate','eval'),{'reports':reports})
    assert low_speed_passed(reports)==old
    assert reports==before
