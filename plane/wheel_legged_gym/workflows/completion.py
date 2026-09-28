"""Terminal report orchestration with an explicitly supplied notification callback."""
import hashlib,json,time
from pathlib import Path
from wheel_legged_gym.ports.notifications import NotificationSink

TERMINAL = frozenset({'finished_pending_review', 'paused_on_regression', 'error', 'stopped',
    'paused_round_budget_review', 'paused_no_improvement_needs_diagnosis',
    'paused_duplicate_experiment_prevented', 'dynamic_source_passed_pending_closed_validation',
    'motion_and_geometry_passed_pending_dynamic_validation',
    'turn_grid_passed_pending_transitions_and_sim2sim',
    'simulation_curriculum_passed_pending_sim2sim_and_smoothness_review'})


def write_report(job,state,notify: NotificationSink):
    """Called by the supervisor once at completion; no LLM or watcher needed."""
    job=Path(job)
    if state['status'] not in TERMINAL:return
    marker=job/'completion_hook.json'
    if marker.exists():return
    rows=[]
    for path in sorted(list(job.glob('*_summary.json')) + list((job/'probes').glob('*_summary.json'))):
        data=json.loads(path.read_text())
        if 'passed_count' in data:
            rows.append({'file':path.name,**{k:data.get(k) for k in [
                'passed_count','total','passed','geometry_passed','max_mean_mirror_distance_m','start_stop_passed']}})
    milestones = state.get('milestones', [])
    last = milestones[-1] if milestones else {}
    candidates = []
    for entry in milestones:
        path = Path(entry.get('probe', ''))
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
            high = data.get('turn_rows', [])[32:44]
            strict = sum(bool(r.get('original_gate', {}).get('passed') and
                              r.get('strict_safety', {}).get('passed')) for r in high)
            slip = max((r.get('worst_environment_slip_rms', float('inf')) for r in high), default=float('inf'))
            candidates.append({'checkpoint': data.get('checkpoint'),
                'checkpoint_sha256': data.get('checkpoint_sha256'),
                'iteration': data.get('checkpoint_iteration', entry.get('checkpoint_iteration')),
                'high_strict_passed': strict, 'high_worst_slip': slip,
                'retention_passed': data.get('retention_passed'),
                'inward_quadrants': data.get('inward_geometry', {}).get('inward_quadrants'),
                'protocol_id':data.get('protocol_id'), 'seed':data.get('seed'),
                'envs_per_command':data.get('envs_per_command'),
                'failed_commands':[row.get('command', 'unknown') for row in data.get('turn_rows', [])
                    if not (row.get('original_gate', {}).get('passed') and
                            row.get('strict_safety', {}).get('passed'))],
                'probe':str(path)})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    best = max(candidates, key=lambda c: (c['retention_passed'] or 0,
        c['inward_quadrants'] or 0, c['high_strict_passed'], -c['high_worst_slip'])) if candidates else None
    event = state.get('event_id') or hashlib.sha256(
        ('training-completion-v1:' + str(job.resolve())).encode()).hexdigest()[:24]
    last_iteration = last.get('checkpoint_iteration', state.get('checkpoint_iteration'))
    last_checkpoint = (str(Path(state['run']) / 'model_{}.pt'.format(last_iteration))
        if state.get('run') and isinstance(last_iteration, int) else None)
    if last_checkpoint and not Path(last_checkpoint).is_file():
        last_checkpoint = None
    context={'schema_version': 2, 'event_id': event, 'job': str(job),
             'status':state['status'],'stop_reason':state.get('stop_reason', 'unknown'),
             'source':state.get('source'),'accepted':state.get('accepted'),
             'error':state.get('error'),'results':rows,'history':state.get('history',[]),
             'last_checkpoint': last_checkpoint,
             'last_checkpoint_iteration': last_iteration,
             'last_checkpoint_sha256': last.get('checkpoint_sha256',
                 state.get('last_checkpoint_sha256')),
             'target_new_iterations': state.get('new_iteration_limit', 'unknown'),
             'actual_update_count': state.get('actual_update_count', 'unknown'),
             'best_candidate': best, 'full_acceptance': 'not_evaluated',
             'milestone_results':candidates,
             'missing': ['three_seed_full_review', 'onnx', 'sim2sim']}
    (job/'terminal_summary.json').write_text(json.dumps(context,indent=2)+'\n')
    (job/'completion_context.json').write_text(json.dumps(context,indent=2)+'\n')
    report=job/'completion_report.md'
    lines=['# 实验结束汇总','',f"job：{job}",f"event_id：{event}",
           f"状态：{state['status']}",f"停止原因：{state.get('stop_reason', 'unknown')}",
           f"来源：{state.get('source')}",
           f"源端接受候选：{state.get('accepted') or '无'}",'',
           '报告生成不代表模型接受；full_acceptance=not_evaluated。','',
           f"目标/实际更新：{state.get('new_iteration_limit', 'unknown')}/{context['actual_update_count']}",
           f"最后保存checkpoint：{last_checkpoint or 'unknown'}",
           f"最后checkpoint SHA：{context['last_checkpoint_sha256'] or 'unknown'}",
           f"最佳单seed筛选候选：{best['checkpoint'] if best else 'not_evaluated'}",'',
           '完整三seed验收：not_evaluated；筛选候选不是已接受模型。','',
           '| 里程碑 | retention | high旧gate+安全 | 最差slip | inward | 证据 |',
           '|---:|---:|---:|---:|---:|---|']
    for candidate in candidates:
        lines.append('| {iteration} | {retention_passed}/5 | {high_strict_passed}/12 | '
            '{high_worst_slip:.4f} | {inward_quadrants}/4 | {probe} |'.format(**candidate))
    if not candidates:
        lines.append('| not_evaluated | - | - | - | - | no completed probe |')
    if best:
        lines += ['', '最佳筛选候选未通过命令：'+str(best['failed_commands']),
                  '评估身份：protocol={}，seed={}，每命令环境数={}'.format(
                      best['protocol_id'],best['seed'],best['envs_per_command'])]
    if rows:
        lines += ['', '| 验收文件 | 运动通过 | 几何通过 | 动态通过 |',
                  '|---|---:|---|---|']
        for row in rows:
            lines.append('| {} | {}/{} | {} | {} |'.format(row['file'],
                row['passed_count'],row['total'],row['geometry_passed'],
                row['start_stop_passed']))
    if state.get('error'):lines+=['',f"错误：{state['error']}"]
    report.write_text('\n'.join(lines)+'\n')
    marker.write_text(json.dumps({'status':'reported','trigger':state['status'],
        'mode':'deterministic_no_llm','event_id':event,'report':str(report),
        'finished_at':time.time()},indent=2)+'\n')
    notify('实验结束，汇总已生成：'+str(report))


def compact_context(job, state):
    result = {'job': str(job), 'status': state['status'], 'error': state.get('error'), 'segments': []}
    for segment in state.get('segments', []):
        item = dict(segment)
        audit = Path(segment['audit']).resolve()
        # Consume only audit files from this job, never arbitrary status paths.
        if audit.parent == job and audit.is_file():
            item['result'] = json.loads(audit.read_text())
        result['segments'].append(item)
    for path in state.get('completed_audits', []):
        audit = Path(path).resolve()
        if audit.parent == job and audit.is_file():
            result['segments'].append({'audit':str(audit),'result':json.loads(audit.read_text())})
    result['source_checkpoint'] = state.get('source_checkpoint')
    return result
