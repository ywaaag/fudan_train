import json
from pathlib import Path
from types import SimpleNamespace

from wheel_legged_gym.adapters.notifications import hapi
from wheel_legged_gym.app import completion
from wheel_legged_gym.app.turn_lean_monitor import alive

from wheel_legged_gym.ports.notifications import NotificationSink
from wheel_legged_gym.workflows.completion import write_report
from wheel_legged_gym.app import wait_for_completion


class Sink:
    def __init__(self): self.messages = []
    def __call__(self, message): self.messages.append(message)


def probe(strict, slip, iteration, checkpoint):
    rows = []
    for i in range(44):
        passed = i < 5
        rows.append({'original_gate': {'passed': passed},
            'strict_safety': {'passed': passed or i in range(32, 32 + strict)},
            'worst_environment_slip_rms': slip, 'yaw_mae': .1})
    return {'checkpoint': checkpoint, 'checkpoint_sha256': 'sha'+str(iteration),
        'checkpoint_iteration': iteration, 'retention_passed': 5,
        'inward_geometry': {'inward_quadrants': 4}, 'turn_rows': rows}


def test_completion_report_selects_screening_candidate_and_terminal_summary(tmp_path):
    job = tmp_path / 'job'; (job / 'probes').mkdir(parents=True)
    for iteration, strict, slip in ((100, 2, .3), (200, 9, .06)):
        (job/'probes'/f'iter_{iteration}_summary.json').write_text(
            json.dumps(probe(strict, slip, iteration, f'/tmp/model_{iteration}.pt')))
    sink = Sink()
    write_report(job, {'status':'finished_pending_review', 'source':'/tmp/source.pt',
        'source_iteration': 0, 'new_iteration_limit': 200,
        'milestones':[{'checkpoint_iteration':100,'probe':str(job/'probes/iter_100_summary.json')},
                      {'checkpoint_iteration':200,'probe':str(job/'probes/iter_200_summary.json')}]}, sink)
    summary = json.loads((job/'terminal_summary.json').read_text())
    assert summary['best_candidate']['iteration'] == 200
    assert summary['full_acceptance'] == 'not_evaluated'
    assert '最佳筛选候选' in (job/'completion_report.md').read_text()
    assert sink.messages


def test_wait_once_json_reads_only_status(tmp_path, capsys):
    (tmp_path/'status.json').write_text(json.dumps({'status':'running','supervisor_pid':99999999}))
    wait_for_completion.main([str(tmp_path), '--once', '--json'])
    assert json.loads(capsys.readouterr().out)['status'] == 'running'


def test_last_checkpoint_is_distinct_from_best_screening_candidate(tmp_path):
    job=tmp_path/'job'; (job/'probes').mkdir(parents=True)
    run=tmp_path/'run';run.mkdir()
    (run/'model_200.pt').write_text('last model')
    for iteration, strict, slip in ((100,9,.04),(200,7,.06)):
        (job/'probes'/f'iter_{iteration}_summary.json').write_text(json.dumps(
            probe(strict,slip,iteration,str(run/f'model_{iteration}.pt'))))
    write_report(job,{'status':'finished_pending_review','event_id':'one-event',
        'source_iteration':0,'new_iteration_limit':200,'actual_update_count':200,
        'run':str(run),'milestones':[
            {'checkpoint_iteration':n,'checkpoint_sha256':'sha'+str(n),
             'probe':str(job/'probes'/f'iter_{n}_summary.json')} for n in (100,200)]},Sink())
    summary=json.loads((job/'terminal_summary.json').read_text())
    assert summary['last_checkpoint']==str(run/'model_200.pt')
    assert summary['best_candidate']['iteration']==100
    assert summary['event_id']=='one-event'
    assert 'not_evaluated' in (job/'completion_report.md').read_text()


def test_persisted_event_is_shared_and_reviewed_delivery_is_not_repeated(tmp_path,monkeypatch):
    job=tmp_path/'attempt/monitor';job.mkdir(parents=True)
    (job.parent/'submit_snapshot.json').write_text(json.dumps({'event_id':'event-one'}))
    (job/'completion_report.md').write_text('ready')
    (job/'report_ready.json').write_text(json.dumps({'event_id':'event-one'}))
    calls=[]
    monkeypatch.setattr(hapi.subprocess,'run',lambda *a,**kw:
        (calls.append(a),SimpleNamespace(returncode=0,stdout='ok'))[1])
    assert hapi.event_id(job,'session')=='event-one'
    hapi.wake_session(job,'session')
    hapi.wake_session(job,'session')
    assert len(calls)==1
    hapi.acknowledge_review(job,'session')
    assert completion.completion_facts(job)['review_status']=='reviewed'
    assert json.loads((job/'hapi_notification.json').read_text())['event_id']=='event-one'


def test_new_job_cannot_notify_before_report_ready(tmp_path,monkeypatch):
    job=tmp_path/'attempt/monitor';job.mkdir(parents=True)
    (job.parent/'submit_snapshot.json').write_text(json.dumps({'event_id':'event-one'}))
    (job/'status.json').write_text(json.dumps({'status':'finished_pending_review'}))
    (job/'completion_report.md').write_text('incomplete')
    monkeypatch.setattr('sys.argv',['hook','--job',str(job),'--once',
        '--hapi-session','session'])
    completion.main()
    assert not (job/'hapi_notification.json').exists()


def test_pid_identity_rejects_reused_pid():
    import os
    from wheel_legged_gym.app.submit_turn_training import process_start
    pid=os.getpid()
    assert alive(pid,process_start(pid))
    assert not alive(pid,process_start(pid)+1)
