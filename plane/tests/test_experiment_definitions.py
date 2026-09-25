"""Static definitions cannot be contaminated by modifying one run manifest."""
import json
import pytest
from wheel_legged_gym.experiments.definitions import EXPERIMENTS, METHOD_V1_PHASES, copy_definition
from wheel_legged_gym.experiments.primitives import apply_legacy_experiment
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO


def test_nested_definitions_are_read_only():
    with pytest.raises(TypeError):EXPERIMENTS['H3']['optimizer']['learning_rate']=1.
    with pytest.raises(TypeError):METHOD_V1_PHASES['stand']['randomization_level']=3
    copy=copy_definition(EXPERIMENTS['H3'])
    assert isinstance(copy['fractions'],tuple)
    copy['optimizer']['learning_rate']=1.
    assert EXPERIMENTS['H3']['optimizer']['learning_rate']==2e-6


def test_mutable_manifest_does_not_change_later_experiment():
    first=apply_legacy_experiment(WheelLeggedCfg(),'H3',WheelLeggedCfgPPO())
    json.dumps(first)
    first['optimizer']['learning_rate']=1.
    second=apply_legacy_experiment(WheelLeggedCfg(),'H3',WheelLeggedCfgPPO())
    assert second['optimizer']['learning_rate']==2e-6
