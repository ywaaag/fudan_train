import isaacgym
from wheel_legged_gym.experiments.recipes.motion_goal import stage_bank


def test_envelope_covers_targets_without_high_speed_tight_turns():
    previous=set(stage_bank('yaw4'))
    for level in range(1,5):
        bank=set(stage_bank('envelope'+str(level)))
        assert previous<=bank
        assert {(-4,0),(4,0),(0,-4),(0,4),(1,-2),(1,2)}<=bank
        for v,w in bank:
            if v< -2:assert w==0
            if abs(v)>2 and w:assert v>0 and abs(v*w)<=3.2+1e-6
            if abs(v)>=3.5:assert abs(w)<=.8
        previous=bank
    assert {(2,1.5),(-2,-1.5),(3,1),(4,.8),(4,-.6)}<=previous
    assert (4,4) not in previous
