"""Explicit low-speed training commands; no measured-state feedback or action shaping."""
import torch


def start_stop_velocity(steps, dt, ramp_seconds=1., amplitude=.5):
    if not 0 < ramp_seconds <= 4.:
        raise ValueError('Ramp duration must lie in (0,4]')
    time=steps.to(dtype=torch.float)*dt
    segment=torch.floor(time/4.).long()%4
    bank=torch.tensor([0.,amplitude,0.,-amplitude],device=steps.device)
    target=bank[segment]
    previous=bank[(segment-1)%4]
    # Every episode starts stationary, without a fictitious negative-speed predecessor.
    previous=torch.where(time<4.,torch.zeros_like(previous),previous)
    fraction=torch.clamp(torch.remainder(time,4.)/ramp_seconds,0.,1.)
    return previous+(target-previous)*fraction


def apply_start_stop(commands, ids, steps, dt, ramp_seconds):
    selected=ids[ids%2==0]
    commands[selected,0]=start_stop_velocity(steps[selected],dt,ramp_seconds)
    commands[selected,1]=0.


class RandomStartStop:
    """Independent per-environment random timers; targets only 0 and +/-speed."""
    def __init__(self,num_envs,device,dt,speed,ramp_seconds,stride=2):
        if stride not in (2,4):raise ValueError('Reviewed cohort stride must be 2 or 4')
        self.dt,self.speed,self.ramp_seconds=dt,speed,ramp_seconds
        self.stride=stride
        self.ids=torch.arange(0,num_envs,stride,device=device)
        self.remaining=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.index=torch.ones(num_envs,device=device,dtype=torch.long)
        self.targets=torch.zeros(num_envs,device=device)

    def durations(self,ids):
        return torch.randint(round(.5/self.dt),round(4./self.dt)+1,(len(ids),),device=ids.device)

    def reset(self,commands,ids):
        ids=ids[ids%self.stride==0]
        self.remaining[ids]=self.durations(ids)
        self.index[ids]=1;self.targets[ids]=0.
        commands[ids,:2]=0.

    def advance(self,commands):
        ids=self.ids
        self.remaining[ids]-=1
        due=ids[self.remaining[ids]<=0]
        self.index[due]=(self.index[due]+torch.randint(1,3,(len(due),),device=ids.device))%3
        self.targets[due]=(self.index[due]-1)*self.speed
        self.remaining[due]=self.durations(due)
        max_delta=self.speed/self.ramp_seconds*self.dt
        commands[ids,0]+=torch.clamp(self.targets[ids]-commands[ids,0],-max_delta,max_delta)
        commands[ids,1]=0.
