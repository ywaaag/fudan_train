"""Acceptance and improvement are different gates; preserve the strict score margin."""
import copy
import pytest
from wheel_legged_gym.workflows.candidate_decision import decide_candidate, recovery_focus


@pytest.mark.parametrize('passed', [False,True])
@pytest.mark.parametrize('score,improved', [(0.48,True),(0.49,False),(0.5,False),(0.8,False)])
def test_acceptance_and_strict_improvement(passed,score,improved):
    result={'passed':passed,'score':score};baseline={'score':.5}
    original=copy.deepcopy((result,baseline))
    decision=decide_candidate(result,baseline)
    assert decision.accepted==passed and decision.improved==improved
    assert decision.history_label==('stage accepted' if passed else
        'improved candidate' if improved else 'rollback retained source')
    assert (result,baseline)==original


@pytest.mark.parametrize('stagnant,repeats', [(0,2),(1,3),(2,4),(5,4)])
def test_focus_retains_order_and_cap(stagnant,repeats):
    baseline={'failed_commands':[(-1.,0.),(0.,1.)]}
    focus=recovery_focus(baseline,stagnant)
    assert focus==[[-1.,0.],[0.,1.]]*repeats
    focus[0][0]=99
    assert baseline['failed_commands'][0]==(-1.,0.)
