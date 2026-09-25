"""Terminal report orchestration with an explicitly supplied notification callback."""
import json,time
from pathlib import Path
from wheel_legged_gym.ports.notifications import NotificationSink

TERMINAL = {'finished_pending_review', 'paused_on_regression', 'error', 'stopped',
    'paused_round_budget_review', 'paused_no_improvement_needs_diagnosis',
    'paused_duplicate_experiment_prevented', 'dynamic_source_passed_pending_closed_validation',
    'motion_and_geometry_passed_pending_dynamic_validation',
    'turn_grid_passed_pending_transitions_and_sim2sim',
    'simulation_curriculum_passed_pending_sim2sim_and_smoothness_review'}


def write_report(job,state,notify: NotificationSink):
    """Called by the supervisor once at completion; no LLM or watcher needed."""
    job=Path(job)
    if state['status'] not in TERMINAL:return
    marker=job/'completion_hook.json'
    if marker.exists():return
    rows=[]
    for path in sorted(job.glob('*_summary.json')):
        data=json.loads(path.read_text())
        if 'passed_count' in data:
            rows.append({'file':path.name,**{k:data.get(k) for k in [
                'passed_count','total','passed','geometry_passed','max_mean_mirror_distance_m','start_stop_passed']}})
    context={'status':state['status'],'source':state.get('source'),'accepted':state.get('accepted'),
             'error':state.get('error'),'results':rows,'history':state.get('history',[])}
    (job/'completion_context.json').write_text(json.dumps(context,indent=2)+'\n')
    report=job/'completion_report.md'
    lines=['# 实验结束汇总','',f"状态：{state['status']}",f"来源：{state.get('source')}",
           f"源端接受候选：{state.get('accepted') or '无'}",'',
           '源端接受不代表闭链通过，也不代表整个goal完成。','',
           '| 验收文件 | 运动通过 | 几何通过 | 动态通过 |',
           '|---|---:|---|---|']
    for r in rows:lines.append(f"| {r['file']} | {r['passed_count']}/{r['total']} | {r['geometry_passed']} | {r['start_stop_passed']} |")
    if state.get('error'):lines+=['',f"错误：{state['error']}"]
    report.write_text('\n'.join(lines)+'\n')
    marker.write_text(json.dumps({'status':'reported','trigger':state['status'],
        'mode':'deterministic_no_llm','report':str(report),'finished_at':time.time()},indent=2)+'\n')
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
