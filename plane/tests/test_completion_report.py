import importlib.util,json
from pathlib import Path

from wheel_legged_gym.app import completion as hook



def test_only_terminal_writes_once_without_llm(tmp_path,monkeypatch):
    notifications=[];monkeypatch.setattr(hook,'notify',notifications.append)
    hook.write_report(tmp_path,{'status':'training'})
    assert not (tmp_path/'completion_hook.json').exists()
    (tmp_path/'r01_m10100_summary.json').write_text(json.dumps({'passed_count':25,'total':25,'passed':False,'geometry_passed':False}))
    state={'status':'paused_round_budget_review','accepted':None}
    hook.write_report(tmp_path,state);hook.write_report(tmp_path,state)
    assert len(notifications)==1
    assert '25/25' in (tmp_path/'completion_report.md').read_text()
    assert json.loads((tmp_path/'completion_hook.json').read_text())['mode']=='deterministic_no_llm'


def test_hapi_delivery_uses_explicit_session_and_sends_once(tmp_path,monkeypatch):
    from types import SimpleNamespace
    (tmp_path/'completion_report.md').write_text('finished')
    calls=[]
    def run(args,**kwargs):
        calls.append((args,kwargs));return SimpleNamespace(returncode=0,stdout='sent')
    monkeypatch.setattr(hook.subprocess,'run',run)
    hook.wake_session(tmp_path,'test-session');hook.wake_session(tmp_path,'test-session')
    assert len(calls)==1
    assert calls[0][0]==['hapi','ping-peer','test-session','--message-file','-']
    assert '只分析结果' in calls[0][1]['input']
    assert json.loads((tmp_path/'hapi_notification.json').read_text())['status']=='sent'


def test_acknowledged_result_does_not_send(tmp_path,monkeypatch):
    (tmp_path/'completion_report.md').write_text('finished')
    hook.acknowledge_review(tmp_path,'session');hook.acknowledge_review(tmp_path,'session')
    def forbidden(*args,**kwargs):raise AssertionError('Already reviewed event must not send')
    monkeypatch.setattr(hook.subprocess,'run',forbidden)
    hook.wake_session(tmp_path,'session')
    assert not (tmp_path/'hapi_notification.json').exists()


def test_concurrent_delivery_is_once(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import time
    (tmp_path/'completion_report.md').write_text('finished');calls=[]
    def run(*args,**kwargs):
        calls.append(1);time.sleep(.03);return SimpleNamespace(returncode=0,stdout='sent')
    monkeypatch.setattr(hook.subprocess,'run',run)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _:hook.wake_session(tmp_path,'session'),range(8)))
    assert len(calls)==1


def test_unknown_delivery_is_not_retried(tmp_path,monkeypatch):
    (tmp_path/'completion_report.md').write_text('finished');calls=[]
    def run(*args,**kwargs):
        calls.append(1);raise hook.subprocess.TimeoutExpired('hapi',90)
    monkeypatch.setattr(hook.subprocess,'run',run)
    hook.wake_session(tmp_path,'session');hook.wake_session(tmp_path,'session')
    assert len(calls)==1
    assert json.loads((tmp_path/'hapi_notification.json').read_text())['status']=='unknown_delivery'
