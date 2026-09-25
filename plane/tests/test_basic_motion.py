import isaacgym
from collections import Counter
from wheel_legged_gym.experiments.recipes.motion_goal import stage_bank


def test_priority_modes_balanced_and_no_combined_turns():
    bank=stage_bank('basic_motion');counts=Counter(bank)
    assert len(bank)==50
    for command in [(0.,0.),(-4.,0.),(4.,0.),(0.,-4.),(0.,4.)]:
        assert counts[command]==6
    assert all(v==0 or w==0 for v,w in bank)
    assert all((-v,-w) in counts for v,w in counts)
    assert {(-.5,0),(.5,0),(-3.5,0),(3.5,0),(0,-2),(0,2)}<=set(bank)
