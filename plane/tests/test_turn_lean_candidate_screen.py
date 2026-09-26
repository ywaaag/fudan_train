"""Candidate screening must not turn missing or incompatible reviews into passes."""
import hashlib
import json

from tools.screen_turn_lean_candidates import load_candidate, main


SEEDS = (19, 37, 53)


def review(tmp_path, label, iteration, *, protocol='frozen protocol',
           train_pass=True, retention_pass=True, turn_height=.38, skill=False):
    root = tmp_path / label
    root.mkdir()
    (root / 'protocol.md').write_text(protocol)
    checkpoint = root / ('model_{}.pt'.format(iteration))
    checkpoint.write_bytes(label.encode())
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (root / 'spec.json').write_text(json.dumps({'stage':{
        'cohort_plan':{'height':.2}} if skill else {}}))
    (root / (label+'_commands.json')).write_text(json.dumps({
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha,
        'spec':str(root/'spec.json')}))
    commands = {'retention':[[float(i),0.,.4] for i in range(25)],
                'turn_train':[[2.5,.6,turn_height]],
                'turn_holdout':[[2.7,.55,turn_height]],
                'entry_exit':[[2.5,.6,turn_height]],
                'slow_reverse':[[2.,.5,turn_height]]}
    if skill:
        commands['height_skill'] = [[vx,0.,height] for height in (.38,.36)
                                    for vx in (0.,-.5,.5)]
        commands['height_entry_exit'] = list(commands['height_skill'])
    points=[]
    point_groups = ('retention','turn_train','turn_holdout') + (('height_skill',) if skill else ())
    for group in point_groups:
        for command in commands[group]:
            passed = (retention_pass if group=='retention' else train_pass)
            points.append({'model':label,'group':group,'command':command,
                'status':'passed' if passed else 'failed','failures':0,
                'max_yaw_mae':.04 if passed else .15,
                'max_slip_rms':.02 if passed else .12,
                'max_height_mae':.005 if passed else .02})
    transitions=[]
    for seed in SEEDS:
        folder=root/'acceptance'/label/('seed'+str(seed))
        folder.mkdir(parents=True)
        for group, group_commands in commands.items():
            (folder/(group+'.json')).write_text(json.dumps({
                'checkpoint':str(checkpoint),'checkpoint_sha256':sha,
                'seed':seed,'schema':'matched_policy_transition_v1',
                'evaluator_sha256':'frozen-evaluator','randomization_level':1,
                'envs_per_command':16,'seconds':18.,'warmup':8.,
                'deterministic':True,'noise':False,
                'results':[{'command':command,'metrics':{'vx_mae':.01}}
                           for command in group_commands]}))
        dynamic_groups = ('entry_exit','slow_reverse') + (('height_entry_exit',) if skill else ())
        for group in dynamic_groups:
            for command in commands[group]:
                transitions.append({'model':label,'seed':seed,'protocol':group,
                    'command':command,'final_phase_passed':True,
                    'failure_count':0})
    (root/'long_evaluation_summary.json').write_text(json.dumps({
        'schema':'turn_lean_long_review_v2',
        'metric_definition_sha256':'fixture-metric-hash',
        'gate_sha256':'fixture-gate-hash',
        'points':points,
        'transitions':transitions}))
    return root


def test_complete_review_uses_all_seeds_and_actual_checkpoint(tmp_path):
    root=review(tmp_path,'early',100)
    row=load_candidate(root)
    assert row['status']=='accepted_review'
    assert row['actual_iteration']==100
    assert row['retention_passed']==25
    assert row['turn_train_passed']==1
    assert row['completed_files']==row['requested_files']==15
    assert row['seeds']==[19,37,53]


def test_later_checkpoint_does_not_outrank_better_earlier_one(tmp_path):
    early=review(tmp_path,'early',100)
    late=review(tmp_path,'late',200,train_pass=False)
    out=tmp_path/'ranking.json'
    main([str(early),str(late),'--out',str(out)])
    ranking=json.loads(out.read_text())
    assert ranking['comparable_count']==2
    assert ranking['recommendation']['actual_iteration']==100
    assert ranking['candidates'][1]['status']=='task_gate_failed'


def test_missing_seed_or_empty_result_is_not_complete(tmp_path):
    root=review(tmp_path,'early',100)
    missing=root/'acceptance/early/seed53/turn_train.json'
    missing.unlink()
    row=load_candidate(root)
    assert row['status']=='incomplete' and not row['eligible']
    assert row['completed_files']==14 and len(row['missing'])==1
    missing.write_text(json.dumps({'results':[]}))
    row=load_candidate(root)
    assert row['status']=='incomplete' and 'metadata/results' in row['reasons'][0]


def test_fixed_height_or_different_protocol_cannot_be_compared(tmp_path):
    fixed=review(tmp_path,'fixed',100,turn_height=.4)
    assert load_candidate(fixed)['status']=='incompatible_fixed_height'
    first=review(tmp_path,'first',200)
    second=review(tmp_path,'second',300,protocol='different gate text')
    out=tmp_path/'comparison.json'
    main([str(first),str(second),'--out',str(out)])
    ranking=json.loads(out.read_text())
    assert ranking['comparable_count']==0
    assert ranking['recommendation'] is None
    assert ranking['candidates'][0]['evaluation_signature'] != ranking['candidates'][1]['evaluation_signature']


def test_retention_regression_and_conflicting_summary_are_rejected(tmp_path):
    root=review(tmp_path,'regressed',100,retention_pass=False)
    assert load_candidate(root)['status']=='retention_regression'
    summary=root/'long_evaluation_summary.json'
    data=json.loads(summary.read_text())
    data['points'][0]['status']='passed'
    data['points'][0]['failures']=1
    summary.write_text(json.dumps(data))
    row=load_candidate(root)
    assert row['status']=='incomplete' and not row['reviewable']


def test_height_skill_requires_its_own_review_groups(tmp_path):
    root=review(tmp_path,'skill',100)
    spec=root/'spec.json'
    spec.write_text(json.dumps({'stage':{'cohort_plan':{'height':.2}}}))
    row=load_candidate(root)
    assert row['status']=='incomplete'
    assert any('height_skill' in path for path in row['missing'])


def test_complete_height_skill_review_can_be_screened(tmp_path):
    root=review(tmp_path,'skill_complete',100,skill=True)
    row=load_candidate(root)
    assert row['status']=='accepted_review'
    assert row['height_skill_passed']==row['height_skill_total']==6
    assert row['transition_total']==24


def test_legacy_metric_identity_and_skipped_point_are_not_accepted(tmp_path):
    root=review(tmp_path,'legacy',100)
    summary=root/'long_evaluation_summary.json'
    data=json.loads(summary.read_text())
    data.pop('metric_definition_sha256')
    data.pop('gate_sha256')
    data['schema']='turn_lean_long_review_v1'
    summary.write_text(json.dumps(data))
    assert load_candidate(root)['status']=='legacy_metric_unknown'
    data['metric_definition_sha256']='fixture-metric-hash'
    data['gate_sha256']='fixture-gate-hash'
    data['points'][25]['status']='unmeasured'
    summary.write_text(json.dumps(data))
    assert load_candidate(root)['status']=='incomplete'


def test_different_command_sets_are_not_compared(tmp_path):
    first=review(tmp_path,'first',100)
    second=review(tmp_path,'second',200)
    raw=second/'acceptance/second/seed19/turn_train.json'
    changed=json.loads(raw.read_text())
    changed['results'][0]['command']=[3.,.7,.38]
    raw.write_text(json.dumps(changed))
    assert load_candidate(second)['status']=='incomplete'
    out=tmp_path/'ranking.json'
    main([str(first),str(second),'--out',str(out)])
    assert json.loads(out.read_text())['comparable_count']==0
