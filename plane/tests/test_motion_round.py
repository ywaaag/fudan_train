"""Exercise promotion boundaries without starting training or exporting policies."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from wheel_legged_gym.workflows.motion_round import RoundMode, RoundProgress, advance_round


MODE = RoundMode(False, False, False, False, False)
SOURCE = Path('/source/model_10000.pt')
CANDIDATE = Path('/candidate/model_10500.pt')


def setup_round():
    baseline = {'score': .5, 'passed': False, 'failed_commands': [[4, 0]]}
    progress = RoundProgress(SOURCE, 0, baseline, 2, [[1, 0]])
    state = {'history': [], 'accepted': str(SOURCE), 'status': 'evaluating',
             'source': str(SOURCE), 'goal_complete': False}
    return progress, state


def forbidden(*args, **kwargs):
    raise AssertionError('Unexpected external operation')


def advance(candidates, progress, state, *, mode=MODE, stages=('basic_motion',),
            evaluate=forbidden, export=forbidden):
    return advance_round(candidates, progress, stages=stages, round_id=3,
                         mode=mode, state=state, evaluate=evaluate, export=export)


def test_no_candidate_retains_source_and_focus():
    progress, state = setup_round()
    updated = advance([], progress, state)
    assert updated==replace(progress, stagnant=3)
    assert state['accepted']==str(SOURCE)
    assert state['history']==[{
        'round': 3, 'stage': 'basic_motion',
        'decision': 'no candidate passed safety and posture retention; retained source',
    }]


@pytest.mark.parametrize('improved', [True, False])
@pytest.mark.parametrize('recover,basic,envelope', [
    (False, False, False), (True, False, False), (True, True, False), (True, False, True),
])
def test_unaccepted_candidate_source_and_focus(improved, recover, basic, envelope):
    progress, state = setup_round()
    result = {'score': .4 if improved else .5, 'passed': False,
              'failed_commands': [[-4, 0]]}
    mode = replace(MODE, recover_motion=recover, basic_motion=basic, speed_envelope=envelope)
    updated = advance([(result['score'], CANDIDATE, result)], progress, state, mode=mode)
    assert updated.source==(CANDIDATE if improved else SOURCE)
    assert updated.stagnant==(0 if improved else 3)
    expected_focus = ([[-4, 0]]*2 if improved else [[4, 0]]*4)
    assert updated.focus==(expected_focus if not recover or basic or envelope else progress.focus)
    assert state['accepted']==str(SOURCE)
    assert state['history'][0]['decision']==(
        'improved candidate' if improved else 'rollback retained source')


@pytest.mark.parametrize('recover,turn,status', [
    (True, False, 'motion_and_geometry_passed_pending_dynamic_validation'),
    (False, True, 'turn_grid_passed_pending_transitions_and_sim2sim'),
    (False, False, 'simulation_curriculum_passed_pending_sim2sim_and_smoothness_review'),
])
def test_final_acceptance_exports_before_recording_and_is_not_goal_completion(recover, turn, status):
    progress, state = setup_round()
    calls = []
    def export(candidate, tag):
        assert state['accepted']==str(SOURCE) and state['history']==[]
        calls.append((candidate, tag))
    result = {'score': .8, 'passed': True}
    updated = advance([(.8, CANDIDATE, result)], progress, state,
                      mode=replace(MODE, recover_motion=recover, turn_curriculum=turn), export=export)
    assert calls==[(CANDIDATE, 'accepted_basic_motion')]
    assert updated.curriculum_finished and updated.stage_index==1
    assert updated.stagnant==0 and updated.focus==[]
    assert state['accepted']==str(CANDIDATE) and state['status']==status
    assert state['source']==str(SOURCE)  # Preserve final journal ordering from supervisor.
    assert state['goal_complete'] is False


@pytest.mark.parametrize('failure_at', ['export', 'enter'])
def test_failure_preserves_publication_boundary(failure_at):
    progress, state = setup_round()
    original = deepcopy(state)
    result = {'score': .4, 'passed': True}
    calls = []
    def export(candidate, tag):
        calls.append('export')
        if failure_at=='export':
            raise RuntimeError('export failed')
    def evaluate(candidate, stage, seeds, tag):
        assert state['accepted']==str(CANDIDATE)
        assert state['history'][0]['decision']=='stage accepted'
        assert (stage, seeds, tag)==('yaw4', [19, 37, 53], 'enter_yaw4')
        calls.append('enter')
        raise RuntimeError('enter failed')
    with pytest.raises(RuntimeError, match=failure_at+' failed'):
        advance([(.4, CANDIDATE, result)], progress, state,
                stages=('basic_motion', 'yaw4'), evaluate=evaluate, export=export)
    assert calls==(['export'] if failure_at=='export' else ['export', 'enter'])
    if failure_at=='export':
        assert state==original


def test_dynamic_rejection_cannot_export_static_pass():
    progress, state = setup_round()
    result = {'score': .4, 'passed': True, 'failed_commands': []}
    calls = []
    def evaluate(candidate, stage, seeds, tag, transition=False):
        calls.append((tag, transition))
        return {'passed': False}
    updated = advance([(.4, CANDIDATE, result)], progress, state,
                      mode=replace(MODE, start_stop_enabled=True), evaluate=evaluate)
    assert calls==[('r03_start_stop', True)]
    assert not updated.curriculum_finished
    assert state['accepted']==str(SOURCE)
    assert state['history'][0]['decision']=='improved candidate'


def test_stage_entry_uses_exported_candidate_and_new_baseline():
    progress, state = setup_round()
    result = {'score': .4, 'passed': True}
    baseline = {'score': .9, 'passed': False}
    calls = []
    def evaluate(candidate, stage, seeds, tag):
        assert calls==['export'] and state['accepted']==str(candidate)
        return baseline
    updated = advance([(.4, CANDIDATE, result)], progress, state,
                      stages=('basic_motion', 'yaw4'), evaluate=evaluate,
                      export=lambda *args: calls.append('export'))
    assert updated==RoundProgress(CANDIDATE, 1, baseline, 0, [])
