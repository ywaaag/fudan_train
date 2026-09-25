import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from command_ramp import ramp_fraction


def test_ramp_starts_zero_holds_exact_target_without_overshoot():
    for shape in ['linear','smoothstep']:
        values=[ramp_fraction(i*.01,2,30,shape) for i in range(5401)]
        assert values[0]==values[200]==0
        assert values[3200]==values[-1]==1
        assert all(a<=b for a,b in zip(values,values[1:]))
    assert ramp_fraction(17,2,30,'smoothstep')==.5
    assert ramp_fraction(2.001,2,30,'smoothstep')<1e-8
