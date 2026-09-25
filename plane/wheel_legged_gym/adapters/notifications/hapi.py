"""HAPI transport and at-most-once event acknowledgement. No training decisions."""
import hashlib,fcntl,json,os,shutil,subprocess,time
from pathlib import Path

def event_id(job,session_id):
    return hashlib.sha256(('training-completion-v1:'+str(Path(job).resolve())+':'+session_id).encode()).hexdigest()[:24]


def acknowledge_review(job,session_id):
    job=Path(job).resolve()
    path=job/'completion_review.json'
    data={'event_id':event_id(job,session_id),'session_id':session_id,'status':'reviewed','acknowledged_at':time.time()}
    try:
        with path.open('x') as f:json.dump(data,f,indent=2)
    except FileExistsError:
        old=json.loads(path.read_text())
        if old.get('event_id')!=data['event_id']:raise RuntimeError('Review acknowledgement belongs to another event')


def wake_session(job,session_id):
    """One delivery attempt per job; preserve ambiguous failures instead of duplicating messages."""
    job=Path(job).resolve()
    with (job/'hapi_notification.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        _wake_session_locked(job,session_id)


def _wake_session_locked(job,session_id):
    identity=event_id(job,session_id)
    ack=job/'completion_review.json'
    if ack.exists() and json.loads(ack.read_text()).get('event_id')==identity:return
    marker=job/'hapi_notification.json'
    if marker.exists():return
    report=job/'completion_report.md'
    if not report.exists():raise RuntimeError('Completion report missing')
    message=(f'训练结束事件 ID：{identity}\n'
        f'请先检查 {ack}：若已记录此事件 reviewed，本次不重复复盘。\n'
        f'训练及验收任务已进入终态：{job}\n'
        f'请读取结束汇总 {report} 和必要的原始结果，复盘是否成功、关键指标及下一步。'
        '这是完成hook，不是成功声明。本次唤醒只分析结果，不修改代码或自动重启训练；'
        '不要只因收到通知就标记goal完成。复盘完成后，仅允许用下面命令写入去重确认记录：\n'
        f'python tools/training_completion_hook.py --job {job} --hapi-session {session_id} --acknowledge-review')
    event={'status':'sending','event_id':identity,'session_id':session_id,'time':time.time()}
    with marker.open('x') as f:json.dump(event,f,indent=2)
    try:
        result=subprocess.run(['hapi','ping-peer',session_id,'--message-file','-'],input=message,
            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        (job/'hapi_notification.log').write_text(result.stdout)
        event.update(status='sent' if result.returncode==0 else 'failed',returncode=result.returncode)
    except (OSError,subprocess.TimeoutExpired) as exc:
        event.update(status='unknown_delivery',error=str(exc))
    marker.write_text(json.dumps(event,indent=2)+'\n')


def notify(message):
    binary = shutil.which('notify-send')
    if binary:
        try:
            subprocess.run([binary, 'Fudan 训练自动复盘', message], timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            pass

