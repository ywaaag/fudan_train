"""Dynamic validation cannot bypass static rejection or reorder gates."""
import pytest
from wheel_legged_gym.workflows.candidate_validation import validate_candidate_transitions


@pytest.mark.parametrize('static_pass', [False, True])
@pytest.mark.parametrize('dynamic_pass', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('stage', ['basic_motion', 'switches'])
def test_dynamic_gate_order_and_acceptance(static_pass, dynamic_pass, enabled, stage):
    result={'passed':static_pass,'score':.4};calls=[]
    def evaluate(candidate, current_stage, seeds, tag, transition=False):
        assert (candidate,current_stage,seeds,transition)==('candidate',stage,[19,37,53],True)
        calls.append(tag)
        return {'passed':dynamic_pass}
    validate_candidate_transitions('candidate',result,stage,3,
                                    start_stop_enabled=enabled,evaluate=evaluate)
    expected=[]
    passed=static_pass
    if enabled:
        expected.append('r03_start_stop');passed=passed and dynamic_pass
        assert result['start_stop']=={'passed':dynamic_pass}
    if passed and stage=='switches':
        expected.append('r03_transition');passed=dynamic_pass
    assert calls==expected and result['passed']==passed
    assert result['score']==.4


def test_error_propagates_before_mutating_result():
    result={'passed':True}
    def evaluate(*args,**kwargs):raise RuntimeError('evaluation failed')
    with pytest.raises(RuntimeError,match='evaluation failed'):
        validate_candidate_transitions('candidate',result,'basic_motion',1,
                                        start_stop_enabled=True,evaluate=evaluate)
    assert result=={'passed':True}
