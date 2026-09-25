"""Same-source control/treatment short training; retain explicit run provenance."""
import json,os,sys,subprocess,time,fcntl
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]


def main():
    lock=(ROOT/'plane/outputs/sim2sim_adaptation_pair.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=ROOT/'plane/outputs'/('sim2sim_adaptation_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    source=ROOT/'plane/logs/wheel_legged/Sep21_14-37-19_motion_goal_20260921_143444_r01_basic_motion/model_9500.pt'
    state={'status':'running','supervisor_pid':os.getpid(),'child_pid':None,'source':str(source),'completed':[],
           'scope':'User permits descendants; original9500 retained. Single-variable command timing control vs treatment.'}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    print(job,flush=True);save()
    try:
        for name,seconds in [('control',0),('switch5',5)]:
            logpath=job/(name+'.log');state['stage']=name
            before=set((ROOT/'plane/outputs').glob('motion_goal_*'))
            args=[sys.executable,str(ROOT/'tools/run_motion_goal.py'),'--recover-motion','--basic-motion',
                  '--recover-from='+str(source),'--command-switch-seconds='+str(seconds),
                  '--training-seed=23','--max-rounds=1','--high-speed-geometry-floor=.03']
            with logpath.open('w') as log:
                child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state.update(command=args,child_pid=child.pid);save()
                childjob=None
                try:
                    while child.poll() is None:
                        # Child prints its absolute output directory after acquiring its training lock.
                        for line in logpath.read_text().splitlines():
                            if line.startswith(str(ROOT/'plane/outputs/motion_goal_')):
                                childjob=Path(line.strip());state['child_job']=str(childjob);save();break
                        if (job/'STOP').exists():
                            if childjob:(childjob/'STOP').touch()
                            else:child.terminate()
                        time.sleep(2)
                    if (job/'STOP').exists():raise InterruptedError('User STOP')
                    if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}')
                finally:
                    if child.poll() is None:
                        if childjob:(childjob/'STOP').touch()
                        else:child.terminate()
                        child.wait()
                    state['child_pid']=None;save()
            if childjob is None:raise RuntimeError('Missing child provenance')
            result=json.loads((childjob/'status.json').read_text())
            state['completed'].append({'stage':name,'job':str(childjob),'status':result['status'],
                                       'history':result['history'],'accepted':result.get('accepted')});save()
        state['status']='training_pair_complete_pending_closed_chain_validation';save()
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc));save();raise


if __name__=='__main__':main()
