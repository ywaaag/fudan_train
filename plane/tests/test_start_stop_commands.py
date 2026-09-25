import isaacgym
import torch
from wheel_legged_gym.domain.commands.start_stop_commands import start_stop_velocity,apply_start_stop
from wheel_legged_gym.domain.commands.start_stop_commands import RandomStartStop


def test_cycle_targets_and_acceleration():
    steps=torch.arange(2001)
    velocity=start_stop_velocity(steps,.01,1.)
    assert torch.allclose(velocity[torch.tensor([0,400,450,500,800,850,900,1200,1250,1300,1600,1700])],
        torch.tensor([0.,0.,.25,.5,.5,.25,0.,0.,-.25,-.5,-.5,0.]),atol=1e-6)
    assert torch.max(torch.abs(velocity[1:]-velocity[:-1]))<.005001


def test_retention_cohort_untouched_and_no_height_change():
    commands=torch.tensor([[4.,0.,.4],[0.,-4.,.4],[-4.,0.,.4],[0.,4.,.4]])
    original=commands.clone()
    apply_start_stop(commands,torch.arange(4),torch.full((4,),450),.01,1.)
    assert torch.equal(commands[1::2],original[1::2])
    assert torch.equal(commands[:,2],original[:,2])
    assert torch.allclose(commands[::2,0],torch.tensor([.25,.25]))
    assert torch.equal(commands[::2,1],torch.zeros(2))


def test_random_schedule_independent_timing_slew_and_retention():
    torch.manual_seed(123)
    commands=torch.zeros((64,3));commands[1::2,0]=4.;commands[:,2]=.4
    scheduler=RandomStartStop(64,'cpu',.01,1.,1.)
    scheduler.reset(commands,torch.arange(64))
    assert scheduler.remaining[::2].min()>=50 and scheduler.remaining[::2].max()<=400
    assert scheduler.remaining[::2].unique().numel()>1
    changed=torch.zeros(32,dtype=torch.bool)
    for _ in range(1000):
        previous=commands.clone();scheduler.advance(commands)
        assert torch.max(torch.abs(commands[::2,0]-previous[::2,0]))<=.010001
        assert torch.equal(commands[1::2],previous[1::2])
        assert torch.all(commands[:,2]==.4)
        changed|=commands[::2,0]!=0
        assert torch.all(torch.isin(scheduler.targets,torch.tensor([-1.,0.,1.])))
    assert changed.all()
    scheduler.reset(commands,torch.tensor([0,1]))
    assert commands[0,0]==0 and commands[1,0]==4


def test_quarter_dynamic_keeps_three_quarters_untouched():
    commands=torch.full((64,3),.4);commands[:,0]=4.
    scheduler=RandomStartStop(64,'cpu',.01,2.,1.,stride=4)
    original=commands.clone();scheduler.reset(commands,torch.arange(64))
    assert len(scheduler.ids)==16
    scheduler.remaining[:]=0;scheduler.advance(commands)
    retained=torch.arange(64)%4!=0
    assert torch.equal(commands[retained],original[retained])
    assert torch.all(commands[~retained,0].abs()<=.020001)
