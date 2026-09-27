import json
import sys

import pytest

from wheel_legged_gym.app import turn_readiness as report_tool
from wheel_legged_gym.app.completion import completion_facts


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_partial_review_never_counts_as_complete():
    groups = {name: {'seeds': [19, 37, 53], 'envs': [16], 'passed': 3,
                     'total': 3} for name in ('retention', 'turn_train',
                     'turn_holdout', 'entry_exit', 'slow_reverse', 'height_skill',
                     'height_entry_exit')}
    review = {'groups': groups, 'protocol': 'inward_stage1_v1',
              'gate': 'gate-sha', 'metric': 'metric-sha', 'sha':'model-sha',
              'identity_complete':True,
              'requested_counts':{name:3 for name in groups}}
    assert report_tool.complete(review)
    review['identity_complete'] = False
    assert not report_tool.complete(review)
    review['identity_complete'] = True
    groups['turn_train']['total'] = 2
    assert not report_tool.complete(review)
    groups['turn_train']['total'] = 3
    assert report_tool.required_groups('inward_stage1_v1') == report_tool.BASE_GROUPS
    review['protocol'] = 'aggressive_cornering_v2'
    assert report_tool.complete(review)
    review['protocol'] = 'inward_stage1_v1'
    del groups['height_skill']
    del groups['height_entry_exit']
    assert report_tool.complete(review)
    review['protocol'] = 'cornering_height_skill_v1'
    assert not report_tool.complete(review)
    groups['height_skill'] = {'seeds':[19,37,53], 'envs':[16], 'passed':3, 'total':3}
    groups['height_entry_exit'] = {'seeds':[19,37,53], 'envs':[16], 'passed':3, 'total':3}
    assert report_tool.complete(review)
    groups['turn_holdout']['seeds'] = [19, 37]
    assert not report_tool.complete(review)
    groups['turn_holdout']['seeds'] = [19, 37, 53]
    groups['turn_holdout']['geometry_missing'] = True
    assert not report_tool.complete(review)
    groups['turn_holdout']['geometry_missing'] = False
    assert report_tool.complete(review)
    groups['turn_train']['strict_safety_missing'] = True
    assert not report_tool.complete(review)
    groups['turn_train']['strict_safety_missing'] = False
    groups['turn_train']['strict_safety_failed'] = True
    assert not report_tool.complete(review)
    groups['turn_train']['strict_safety_failed'] = False
    review['protocol'] = 'legacy_unknown'
    assert not report_tool.complete(review)
    review['protocol'] = 'cornering_height_skill_v1'
    review['protocol'] = None
    assert not report_tool.complete(review)


def test_actual_checkpoint_budget_and_missing_height_are_explicit(tmp_path):
    root = tmp_path / 'experiment'
    job = root / 'stage2_job'
    run = root / 'run'
    write(run / 'policy_experiment.json', {'turn_long_spec':{'stage':{'cohort_plan':
        {'height_bank':[[0.,.38]]}}}})
    write(root / 'budget_ledger.json', {'total_limit': 25000,
        'entries': [{'actual_new_iterations': 7050},
                    {'actual_new_iterations': 2000}]})
    write(root / 'spec_stage2_height.json', {'source_iteration': 16250})
    write(job / 'status.json', {'status': 'finished_pending_review',
        'stage': 2, 'latest_iteration': 17250, 'run':str(run)})
    write(job / 'probes/iter_17250_summary.json', {
        'target_iteration': 17200, 'checkpoint': '/run/model_17250.pt',
        'checkpoint_sha256': 'sha', 'retention_passed': 5,
        'retention_total': 5, 'mid_rows': [{'min_wheel_contact': 1.,
        'slip_rms':0.,'failure_count': 0}]})
    status, probe, reviews, accepted, budget, readiness = report_tool.build(job)
    assert budget['used'] == 10050 and budget['remaining'] == 14950
    assert readiness['checkpoint_iteration'] == 17250
    assert readiness['decisions']['low_speed_height_skill'] == 'insufficient_evidence'
    assert readiness['decisions']['final_acceptance'] == 'insufficient_evidence'
    assert 'height_0p38_probe' in readiness['missing']
    assert not reviews and not accepted


def test_original_gate_cannot_bypass_strict_contact(tmp_path):
    job = tmp_path / 'stage1_job'
    write(job / 'status.json', {'status':'finished_pending_review',
        'stage':1,'latest_iteration':16250})
    write(job / 'probes/iter_16250_summary.json', {
        'checkpoint':'/run/model_16250.pt','checkpoint_sha256':'sha',
        'retention_passed':5,'retention_total':5,
        'turn_rows':[{'original_gate':{'passed':True},
            'min_wheel_contact':.6,'slip_rms':.01,'failure_count':0}]})
    *_, readiness = report_tool.build(job)
    assert readiness['protocol_id'] == 'inward_stage1_v1'
    assert 'height_skill' not in readiness['required_groups']
    assert readiness['decisions']['low_speed_height_skill'] != 'ready'
    assert 'turn_contact_probe' in readiness['failed']


def test_aggressive_v2_requires_review_and_report_separates_last_from_best(tmp_path):
    job = tmp_path / 'stage2_job'
    run = tmp_path / 'run'
    write(run/'policy_experiment.json', {'turn_long_spec':{'stage':{
        'aggressive_plan':{'sampling_version':2}}}})
    write(job/'status.json', {'status':'finished_pending_review','stage':2,
        'latest_iteration':24250,'run':str(run)})
    write(job/'probes/iter_24250_summary.json', {'checkpoint':'/run/model_24250.pt',
        'checkpoint_sha256':'last-sha','retention_passed':5,'retention_total':5,
        'turn_rows':[{'min_wheel_contact':1.,'slip_rms':0.,'failure_count':0}]})
    status,probe,reviews,accepted,budget,readiness = report_tool.build(job)
    assert readiness['protocol_id'] == 'aggressive_cornering_v2'
    assert 'height_skill' not in readiness['required_groups']
    assert readiness['decisions']['final_acceptance'] == 'insufficient_evidence'
    assert readiness['recommendation'] is None
    accepted = [{'checkpoint':'/run/model_16250.pt','path':'/review/early',
        'protocol':'aggressive_cornering_v2','groups':{}}]
    text = report_tool.report(job,status,probe,reviews,accepted,budget,readiness)
    assert 'model_24250.pt' in text and 'model_16250.pt' in text


def test_readiness_prefers_v3_and_rejects_legacy_identity(tmp_path):
    job = tmp_path/'stage1_job'
    review = tmp_path/'phase_reviews'/'candidate'
    write(review/'candidate_commands.json', {'checkpoint':'/run/model_16250.pt',
        'checkpoint_sha256':'sha','commands':[]})
    write(review/'long_evaluation_summary.json', {
        'schema':'turn_lean_long_review_v2','inward_protocol':'inward_stage1_v1',
        'gate_sha256':'gate','metric_definition_sha256':'metric'})
    found = report_tool.review_rows(job)
    assert len(found) == 1 and not found[0]['identity_complete']
    write(review/'long_evaluation_summary_v3.json', {
        'schema':'turn_lean_long_review_v3','protocol_id':'inward_stage1_v1',
        'checkpoint_sha256':'sha','gate_id':'gate',
        'metric_definition_sha256':'metric','missing_files':[],
        'evaluation_records':[{'checkpoint_sha256':'sha',
            'evaluator_sha256':'evaluator','completed':True}]})
    found = report_tool.review_rows(job)
    assert len(found) == 1 and found[0]['identity_complete']
    assert found[0]['path'].endswith('long_evaluation_summary_v3.json')


def test_readiness_does_not_compare_different_gate_or_evaluator_versions():
    first = {'identity_complete':True,'protocol':'aggressive_cornering_v2',
             'gate':'gate-a','metric':'metric-a','evaluator':'eval-a'}
    assert report_tool.same_review_identity([first])
    for field,value in [('protocol','inward_stage1_v1'),('gate','gate-b'),
                        ('metric','metric-b'),('evaluator','eval-b')]:
        other = dict(first,**{field:value})
        assert not report_tool.same_review_identity([first,other])


def test_report_error_does_not_rewrite_terminal_status(tmp_path, monkeypatch):
    job = tmp_path / 'stage2_job'
    status = {'status': 'finished_pending_review', 'latest_iteration': 17250}
    write(job / 'status.json', status)
    monkeypatch.setattr(report_tool, 'report', lambda *args: 1 / 0)
    monkeypatch.setattr(sys, 'argv', ['report', '--job', str(job),
        '--out-dir', str(tmp_path / 'sidecar')])
    with pytest.raises(ZeroDivisionError):
        report_tool.main()
    assert json.loads((job / 'status.json').read_text()) == status
    assert json.loads((tmp_path / 'sidecar/report_error.json').read_text())[
        'source_status'] == 'finished_pending_review'


def test_old_completion_records_are_read_without_resending(tmp_path):
    write(tmp_path / 'completion_hook.json', {'status': 'reported'})
    (tmp_path / 'completion_report.md').write_text('report')
    write(tmp_path / 'hapi_notification.json', {'event_id': 'event-1',
        'status': 'sent'})
    write(tmp_path / 'completion_review.json', {'event_id': 'event-1',
        'status': 'reviewed', 'acknowledged_at': 42})
    facts = completion_facts(tmp_path)
    assert facts['report_generated'] and facts['notification_status'] == 'sent'
    assert facts['review_status'] == 'reviewed' and facts['reviewed_at'] == 42
    assert facts['event_id'] == 'event-1'
    assert facts['report_path'] == str(tmp_path / 'completion_report.md')
    assert facts['status'] == 'reported'
    write(tmp_path / 'completion_review.json', {'event_id': 'other',
        'status': 'reviewed'})
    assert completion_facts(tmp_path)['review_status'] == 'unknown'
    write(tmp_path / 'hapi_notification.json', {'event_id':'event-1',
        'session_id':'one','status':'sent'})
    write(tmp_path / 'completion_review.json', {'event_id':'event-1',
        'session_id':'two','status':'reviewed'})
    assert completion_facts(tmp_path)['session_id'] == 'unknown'
