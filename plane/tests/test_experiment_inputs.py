"""Explicit recipes and legacy process inputs must produce identical configs."""
import ast
import json
from pathlib import Path

import pytest

from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment as legacy_apply
from wheel_legged_gym.contracts.config_serialization import class_to_dict
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.experiments.selection import apply_policy_experiment


@pytest.mark.parametrize("name,variable,spec", [
    ("MOTION_GOAL", "FUDAN_MOTION_GOAL_SPEC", {"stage": "basic_motion", "geometry_symmetry": True}),
    ("MOTION_GOAL", "FUDAN_MOTION_GOAL_SPEC", {"stage": "yaw4", "focus": [[0, 4]]}),
    ("HEIGHT_COURSE", "FUDAN_HEIGHT_SPEC", {"stage": "micro", "height_switch_interval": 5}),
    ("HEIGHT_COURSE", "FUDAN_HEIGHT_SPEC", {"stage": "dual", "height_reward_gain": 4}),
])
def test_explicit_recipe_ignores_process_environment(tmp_path, monkeypatch, name, variable, spec):
    monkeypatch.setenv(variable, str(tmp_path / "does-not-exist.json"))
    explicit, explicit_train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    manifest = apply_policy_experiment(explicit, name, explicit_train, spec=spec)
    # A bad ambient path must not affect an explicitly supplied specification.
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setenv(variable, str(path))
    legacy, legacy_train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    legacy_manifest = legacy_apply(legacy, name, legacy_train)
    assert manifest == legacy_manifest
    assert class_to_dict(explicit) == class_to_dict(legacy)
    assert class_to_dict(explicit_train) == class_to_dict(legacy_train)


def test_train_resume_does_not_import_removed_utils_exports():
    root = Path(__file__).parents[1] / "wheel_legged_gym"
    tree = ast.parse((root / "scripts/train.py").read_text())
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "wheel_legged_gym.utils"
                   for node in ast.walk(tree))


def test_invalid_focus_rejected_by_explicit_recipe():
    with pytest.raises(ValueError, match="Focus must retain"):
        apply_policy_experiment(WheelLeggedCfg(), "MOTION_GOAL", WheelLeggedCfgPPO(),
                                spec={"stage": "yaw05", "focus": [[4, 4]]})


@pytest.mark.parametrize('level', ['0', '1', '2', '3'])
def test_stand_randomization_is_explicit_and_legacy_compatible(monkeypatch, level):
    from wheel_legged_gym.experiments.primitives import apply_training_profile
    from wheel_legged_gym.app.experiment_inputs import apply_training_profile as legacy
    monkeypatch.setenv('FUDAN_STAND_RANDOMIZATION_LEVEL', 'invalid ambient value')
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    manifest = apply_training_profile(cfg, train, stand_randomization_level=level)
    monkeypatch.setenv('FUDAN_STAND_RANDOMIZATION_LEVEL', level)
    other, other_train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    assert legacy(other, other_train) == manifest
    assert class_to_dict(cfg) == class_to_dict(other)
    assert class_to_dict(train) == class_to_dict(other_train)


def test_nonstand_ignores_invalid_legacy_stand_override(monkeypatch):
    from wheel_legged_gym.app.experiment_inputs import apply_training_profile
    monkeypatch.setenv('FUDAN_STAND_RANDOMIZATION_LEVEL', 'invalid')
    apply_training_profile(WheelLeggedCfg(), WheelLeggedCfgPPO(), phase='translate')
    with pytest.raises(ValueError):
        apply_training_profile(WheelLeggedCfg(), WheelLeggedCfgPPO(), phase='stand')


def test_recipes_never_read_environment_variables():
    root = Path(__file__).parents[1] / 'wheel_legged_gym/experiments'
    for path in root.rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {'environ', 'getenv'}, str(path)


def test_recipes_do_not_own_files_or_checkpoint_loading():
    root = Path(__file__).parents[1] / 'wheel_legged_gym/experiments'
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {'read_text', 'read_bytes', 'write_text',
                                                  'write_bytes', 'load', 'save'}, str(path)
                if isinstance(node.func, ast.Name):
                    assert node.func.id != 'open', str(path)
