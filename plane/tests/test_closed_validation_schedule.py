import importlib.util
from pathlib import Path
import threading,time
import pytest
spec=importlib.util.spec_from_file_location('schedule',Path(__file__).resolve().parents[2]/'tools/closed_validation_schedule.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_parallel_retains_all_cases_and_same_physical_failure_gates():
    outputs=[]
    for workers in [1,3]:
        rows=[];skips=[]
        def evaluate(case):return {'tag':case[0],'physical':case[0]!='vx_-1_1'}
        module.run_grid(evaluate,rows.append,skips.append,workers)
        outputs.append(({r['tag'] for r in rows},set(skips)))
    assert outputs[0]==outputs[1]
    assert len(outputs[0][0])==11 and outputs[0][1]=={'vx_-1_2','vx_-1_4'}


def test_concurrency_is_bounded_and_zero_completes_first():
    lock=threading.Lock();active=0;peak=0;zero=False;rows=[]
    def evaluate(case):
        nonlocal active,peak,zero
        if case[0]=='zero':zero=True;return {'tag':'zero','physical':True}
        assert zero
        with lock:active+=1;peak=max(peak,active)
        time.sleep(.01)
        with lock:active-=1
        return {'tag':case[0],'physical':True}
    module.run_grid(evaluate,rows.append,lambda x:None,workers=2)
    assert peak==2 and len(rows)==13


def test_process_error_cancels_without_starting_higher_speed():
    called=[];cancelled=[]
    def evaluate(case):
        called.append(case[0])
        if case[0]=='vx_1_1':raise RuntimeError('child process failed')
        return {'tag':case[0],'physical':True}
    with pytest.raises(RuntimeError):
        module.run_grid(evaluate,lambda r:None,lambda t:None,workers=2,cancel=lambda:cancelled.append(True))
    assert cancelled==[True]
    assert all(not tag.endswith(('_2','_4')) for tag in called)
