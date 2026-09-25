import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('command_sequence',Path(__file__).resolve().parents[2]/'tools/command_sequence.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_holds_ramps_and_exact_step():
    points=[{'time':0,'command':[0,0,.4]},{'time':2,'command':[0,0,.4]},
            {'time':3,'command':[1,0,.4]},{'time':5,'command':[1,0,.4]},
            {'time':6,'command':[0,0,.4]}]
    sequence=module.CommandSequence({'keyframes':points})
    assert sequence.at(1)==[0,0,.4]
    assert sequence.at(2.5)==[.5,0,.4]
    assert sequence.at(5.5)==[.5,0,.4]
    assert sequence.at(100)==[0,0,.4]
    step=module.CommandSequence({'keyframes':points,'interpolation':'step'})
    assert step.at(2.999)==[0,0,.4] and step.at(3)==[1,0,.4]


def test_rejects_ambiguous_or_changed_height():
    with pytest.raises(ValueError):
        module.CommandSequence({'keyframes':[{'time':0,'command':[0,0,.4]},{'time':0,'command':[1,0,.4]}]})
    with pytest.raises(ValueError):
        module.CommandSequence({'keyframes':[{'time':0,'command':[0,0,.4]},{'time':2,'command':[1,0,.3]}]})
