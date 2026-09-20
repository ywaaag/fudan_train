import isaacgym
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from wheel_legged_gym.envs.wheel_legged.motion_goal import STAGES,stage_bank
from run_motion_goal import assess


def test_all_stages_retain_previous_commands_and_reach_targets():
    previous=set()
    for stage in STAGES:
        commands=set(stage_bank(stage))
        assert previous <= commands
        assert {(0,0),(-2,0),(2,0),(0,-.5),(0,.5)} <= commands
        previous=commands
    assert {(-4,0),(4,0),(0,-4),(0,4)} <= previous


def test_turn_curriculum_retains_motion_and_progresses_both_directions():
    previous=set(stage_bank('yaw4'))
    for i in range(1,5):
        current=set(stage_bank('turn'+str(i)))
        assert previous<=current
        assert {(s*float(i),t*.5) for s in (-1,1) for t in (-1,1)}<=current
        assert {(0,0),(-4,0),(4,0),(0,-4),(0,4)}<=current
        previous=current
    assert {(-4,-.25),(-4,.25),(4,-.25),(4,.25)}<=previous


def test_unsafe_policy_cannot_be_promoted_even_with_zero_error():
    row=dict(command=[0,.5,.4],failure_count=1,timeout_count=0,
             nonwheel_contact_full_fraction=0,
             metrics=dict(vx_mae=0,yaw_mae=0,abs_yaw=.5,nonwheel_contact=0,
                          left_contact=1,right_contact=1,height_mae=0,torque_saturation=0))
    score=assess([{'results':[row]}])
    assert not score['safe'] and not score['passed']
    row['failure_count']=0
    assert assess([{'results':[row]}])['passed']
    row['nonwheel_contact_full_fraction']=.01
    assert not assess([{'results':[row]}])['safe']
