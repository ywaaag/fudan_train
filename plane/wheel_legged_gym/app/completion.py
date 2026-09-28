"""Completion composition root: joins reporting and transport without reverse dependencies."""
import argparse,fcntl,json,os,shutil,subprocess,time
from pathlib import Path
from wheel_legged_gym.workflows.completion import TERMINAL, compact_context
from wheel_legged_gym.workflows.completion import write_report as generate_report
from wheel_legged_gym.adapters.notifications.hapi import event_id, acknowledge_review, wake_session, notify


def completion_facts(job):
    """Normalize independent report, delivery and human-review records."""
    job = Path(job)
    def read(name):
        path = job / name
        return json.loads(path.read_text()) if path.is_file() else {}
    marker = read('completion_hook.json')
    delivery = read('hapi_notification.json')
    review = read('completion_review.json')
    event = delivery.get('event_id') or review.get('event_id')
    reviewed = (review.get('status') == 'reviewed' and
                bool(event) and review.get('event_id') == event)
    report_path = Path(marker.get('report', job / 'completion_report.md'))
    if not report_path.is_absolute():
        report_path = job / report_path
    report_generated = marker.get('status') == 'reported' and report_path.is_file()
    delivery_session = delivery.get('session_id')
    review_session = review.get('session_id')
    session = (delivery_session or review_session) if not (
        delivery_session and review_session and delivery_session != review_session) else None
    return {'event_id': event or 'unknown',
            'status': marker.get('status', 'unknown'),
            'session_id': session or 'unknown',
            'report_path': str(report_path) if report_generated else 'unknown',
            'report_generated': report_generated,
            'notification_status': delivery.get('status', 'unknown'),
            'review_status': 'reviewed' if reviewed else 'unknown',
            'reviewed_at': review.get('acknowledged_at', 'unknown') if reviewed else 'unknown'}


def write_report(job,state):
    return generate_report(job,state,notify=notify)

def review(job, state, codex):
    """Generate the deterministic report; the interactive session does the review."""
    marker = job / 'completion_hook.json'
    if marker.exists():
        return
    write_report(job, state)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--codex', default=shutil.which('codex'))
    p.add_argument('--poll-seconds', type=float, default=30)
    p.add_argument('--once', action='store_true', help='Check once; suitable for cron or tests')
    p.add_argument('--report-only',action='store_true',help='Generate deterministic report without another Codex session')
    p.add_argument('--hapi-session',help='Explicit existing HAPI session to wake once after terminal report')
    p.add_argument('--acknowledge-review',action='store_true',help='Record completed analysis without sending a notification')
    args = p.parse_args()
    if args.acknowledge_review:
        if not args.hapi_session:p.error('Acknowledgement requires explicit HAPI session')
        acknowledge_review(args.job.resolve(strict=True),args.hapi_session);return
    if args.poll_seconds < 1:
        p.error('poll interval must be >= 1')
    job = args.job.resolve(strict=True)
    with (job / 'completion_hook.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        while True:
            state = json.loads((job / 'status.json').read_text())
            if state['status'] in TERMINAL:
                if (job.parent/'submit_snapshot.json').is_file() and not (
                        job/'report_ready.json').is_file():
                    if args.once:return
                    time.sleep(args.poll_seconds)
                    continue
                if args.report_only:write_report(job,state)
                else:review(job, state, args.codex)
                session_id = (args.hapi_session if (job.parent/'submit_snapshot.json').is_file()
                              else args.hapi_session or os.environ.get('HAPI_SESSION_ID'))
                if session_id:
                    wake_session(job,session_id)
                return
            if args.once:
                return
            time.sleep(args.poll_seconds)
