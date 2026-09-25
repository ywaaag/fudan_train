"""Legacy class-based defaults must not leak mutations between experiment instances."""
from wheel_legged_gym.contracts.base_config import BaseConfig
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.contracts.config_serialization import class_to_dict


def test_nested_mutable_defaults_are_owned_by_each_instance():
    class Config(BaseConfig):
        class section:
            values = [1, {'nested': [2]}]
            mapping = {'x': [3]}
            sequence = ([4],)
    first, second = Config(), Config()
    baseline = class_to_dict(second)
    first.section.values[1]['nested'].append(5)
    first.section.mapping['x'].clear()
    first.section.sequence[0].append(6)
    assert class_to_dict(second) == baseline
    assert class_to_dict(Config()) == baseline
    assert Config.section.values == [1, {'nested': [2]}]


def test_robot_defaults_remain_identical_and_isolated():
    a, b = WheelLeggedCfg(), WheelLeggedCfg()
    baseline = class_to_dict(b)
    a.init_state.pos[2] += .1
    a.init_state.default_joint_angles.clear()
    assert class_to_dict(b) == baseline
    assert class_to_dict(WheelLeggedCfg()) == baseline
    x, y = WheelLeggedCfgPPO(), WheelLeggedCfgPPO()
    training_baseline = class_to_dict(y)
    x.policy.actor_hidden_dims.append(7)
    assert class_to_dict(y) == training_baseline
    assert class_to_dict(WheelLeggedCfgPPO()) == training_baseline
