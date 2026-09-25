"""Screening must preserve eligibility, ties, call sequence and second-pass rejection."""
from pathlib import Path
import pytest

from wheel_legged_gym.workflows.candidate_screening import screen_candidates


def test_screening_order_ties_and_second_pass_safety():
    calls=[]
    def evaluate(checkpoint,stage,seeds,tag):
        iteration=int(checkpoint.stem.split('_')[-1]);calls.append((iteration,seeds,tag))
        assert stage=='basic_motion'
        result=dict(safe=True,score=1.)
        if iteration==1100:result['safe']=False
        if iteration==1200:result['posture_retained']=False
        if len(seeds)==3 and iteration==1300:result['safe']=False
        return result
    result=screen_candidates(Path('/run'),1000,'basic_motion',2,evaluate=evaluate)
    assert calls==[(i,[19],f'r02_m{i}') for i in range(1100,1501,100)]+[
        (1300,[19,37,53],'r02_m1300'),(1400,[19,37,53],'r02_m1400')]
    assert len(result)==1 and result[0][1]==Path('/run/model_1400.pt')


def test_no_safe_candidates_never_runs_second_pass():
    calls=[]
    def evaluate(checkpoint,stage,seeds,tag):
        calls.append(seeds);return dict(safe=False,score=0.)
    assert screen_candidates(Path('/run'),0,'stage',1,evaluate=evaluate)==[]
    assert calls==[[19]]*5


def test_evaluation_error_propagates_without_more_candidates():
    calls=[]
    def evaluate(*args):
        calls.append(args);raise RuntimeError('evaluation failed')
    with pytest.raises(RuntimeError,match='evaluation failed'):
        screen_candidates(Path('/run'),0,'stage',1,evaluate=evaluate)
    assert len(calls)==1
