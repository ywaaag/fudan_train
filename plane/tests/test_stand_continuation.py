"""Compare extracted gates with the unchanged pre-refactor supervisor fixture."""
import ast
import copy
import json
from pathlib import Path

import pytest

from wheel_legged_gym.evaluation.stand_continuation import assess_segment


def original_gate():
    source = json.loads((Path(__file__).parent/'fixtures/legacy_supervisor_bodies.json').read_text())
    tree = ast.parse(source['stand_long_guard'])
    healthy = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'healthy')
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'segment')
    start = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='reasons')
    end = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr)
               and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
               and n.value.func.attr=='append')
    # File loading stays in app; supply the same parsed audits explicitly.
    body = [healthy, loop.body[start]] + loop.body[start+2:end]
    return compile(ast.Module(body=body,type_ignores=[]), 'frozen_stand_gate', 'exec')


def report(seed):
    return {'seed':seed, 'failure_count':0, 'timeout_count':0,
            'metrics':{'nonwheel_contact_fraction':{'mean':0},
                       'left_contact':{'mean':1}, 'right_contact':{'mean':1},
                       'height_m':{'mean':.4}, 'roll_rad':{'mean_abs':0}, 'pitch_rad':{'mean_abs':0}},
            'push_results':[{'recovery_seconds':[2.]*160}],
            'geometry_root_frame':{'knee':{'mean_distance_m':.01}, 'wheel':{'mean_distance_m':.02}},
            'position_drift':{'final_displacement_mean_m':.1}}


@pytest.mark.parametrize('case', ['pass', 'missing', 'duplicate', 'failure', 'timeout',
    'contact', 'height', 'roll', 'none_recovery', 'late_recovery', 'few_trials',
    'geometry_equal', 'geometry_over', 'drift_equal', 'drift_over', 'combined'])
def test_gate_matches_original(case):
    baseline = {seed:report(seed) for seed in [19,37,53]}
    audits = copy.deepcopy(list(baseline.values()))
    current = report(19)
    base_stationary = report(19)
    if case=='missing': audits.pop()
    if case=='duplicate': audits[1]['seed']=19
    if case=='failure': audits[0]['failure_count']=1
    if case=='timeout': audits[0]['timeout_count']=1
    if case=='contact': audits[0]['metrics']['left_contact']['mean']=.989
    if case=='height': current['metrics']['height_m']['mean']=.5
    if case=='roll': current['metrics']['roll_rad']['mean_abs']=.10
    if case=='none_recovery': audits[0]['push_results'][0]['recovery_seconds'][0]=None
    if case=='late_recovery': audits[0]['push_results'][0]['recovery_seconds'][0]=2.0001
    if case=='few_trials': audits[0]['push_results'][0]['recovery_seconds'].pop()
    if case.startswith('geometry'):
        audits[0]['geometry_root_frame']['knee']['mean_distance_m']=.01*1.2+.002+(1e-9 if case.endswith('over') else 0)
    if case.startswith('drift'):
        current['position_drift']['final_displacement_mean_m']=.1*1.2+.05+(1e-9 if case.endswith('over') else 0)
    if case=='combined':
        audits[0]['failure_count']=1
        audits[0]['push_results']=[]
        audits[0]['geometry_root_frame']['wheel']['mean_distance_m']=1.
        current['failure_count']=1
        current['position_drift']['final_displacement_mean_m']=1.
    inputs = dict(audits=audits,base_audits=baseline,current=current,base_stationary=base_stationary)
    before = copy.deepcopy(inputs)
    namespace = dict(inputs)
    exec(original_gate(),namespace)
    assert assess_segment(**inputs)==(namespace['reasons'],namespace['drift'])
    assert inputs==before
