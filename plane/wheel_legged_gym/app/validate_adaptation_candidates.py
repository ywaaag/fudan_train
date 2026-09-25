"""Wait for the paired training, then validate only source-accepted ONNX candidates."""
import json,sys,os,time,subprocess,fcntl,argparse
from pathlib import Path



def main(root):
    ROOT = Path(root)
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('pair',type=Path)
    parser.add_argument('--single-job',action='store_true',help='Watch one basic-motion training job instead of a pair')
    args=parser.parse_args()
    pair=args.pair.resolve();lock=(pair/'closed_validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=pair/'closed_validation_status.json'
    if out.exists():raise FileExistsError(out)
    state={'status':'waiting_for_training','supervisor_pid':os.getpid(),'child_pid':None,'results':[]}
    def save():
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out)
    save()
    try:
        while True:
            if (pair/'STOP').exists():raise InterruptedError('User STOP')
            training=json.loads((pair/'status.json').read_text())
            if args.single_job and (training['status'].startswith('paused_') or
                    training['status']=='motion_and_geometry_passed_pending_dynamic_validation'):
                training={'completed':[{'stage':'basic_motion','job':str(pair),
                                        'accepted':training.get('accepted')}]}
                break
            if training['status']=='training_pair_complete_pending_closed_chain_validation':break
            if training['status'] in ['error','stopped']:raise RuntimeError('Training did not finish: '+str(training.get('error')))
            os.kill(training['supervisor_pid'],0)
            time.sleep(2)
        for entry in training['completed']:
            if not entry.get('accepted'):
                state['results'].append({'stage':entry['stage'],'status':'not_run_source_acceptance_failed'});save();continue
            childjob=Path(entry['job']);policy=childjob/'accepted_basic_motion.onnx'
            if not policy.exists():raise FileNotFoundError(policy)
            with (pair/(entry['stage']+'_closed.log')).open('w') as log:
                child=subprocess.Popen(['/home/kellen/anaconda3/envs/robot/bin/python',str(ROOT/'tools/validate_closed_ramp.py'),
                    '--policy',str(policy)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state.update(status='validating',stage=entry['stage'],child_pid=child.pid);save()
                validation_job=None
                try:
                    while child.poll() is None:
                        for line in (pair/(entry['stage']+'_closed.log')).read_text().splitlines():
                            if line.startswith(str(ROOT/'plane/outputs/closed_ramp_v2_')):
                                validation_job=Path(line.strip());state['validation_job']=str(validation_job);save()
                        if (pair/'STOP').exists():
                            if validation_job:(validation_job/'STOP').touch()
                            else:child.terminate()
                        time.sleep(2)
                    if (pair/'STOP').exists():raise InterruptedError('User STOP')
                    if child.returncode:raise RuntimeError('Closed validation process failed')
                finally:
                    if child.poll() is None:
                        if validation_job:(validation_job/'STOP').touch()
                        else:child.terminate()
                        child.wait()
                    state['child_pid']=None;save()
            if not validation_job:raise RuntimeError('Missing validation provenance')
            result=json.loads((validation_job/'status.json').read_text())
            state['results'].append({'stage':entry['stage'],'policy':str(policy),'job':str(validation_job),
                                    'results':result['results'],'skipped':result['skipped']});save()
        state['status']='completed_pending_comparison';save()
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc));save();raise
