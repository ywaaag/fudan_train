"""Resolve legacy environment/file inputs before invoking explicit recipes.

Only application entrypoints call this module. Pure recipe clients pass spec
dictionaries to experiments directly, independent of the process environment.
"""
import json
import os
from pathlib import Path

from wheel_legged_gym.experiments import selection
from wheel_legged_gym.experiments import primitives
from wheel_legged_gym.experiments.recipes import motion_goal, height_course
from wheel_legged_gym.adapters.artifacts import recipe_source


def read_motion_spec():
    spec = json.loads(Path(os.environ['FUDAN_MOTION_GOAL_SPEC']).read_text())
    return motion_goal.validate_spec(spec)


def read_height_spec():
    return json.loads(Path(os.environ['FUDAN_HEIGHT_SPEC']).read_text())


def read_turn_spec():
    from wheel_legged_gym.experiments.recipes.turn_envelope import validate_spec
    return validate_spec(json.loads(Path(os.environ['FUDAN_TURN_SPEC']).read_text()))


def apply_motion_goal(cfg, train):
    return motion_goal.apply_motion_goal(cfg, train, spec=read_motion_spec())


def apply_height_course(cfg, train):
    return height_course.apply_height_course(cfg, train, spec=read_height_spec())


def apply_policy_experiment(env_cfg, name, train_cfg=None):
    spec = None
    if str(name).upper() == 'MOTION_GOAL':
        spec = read_motion_spec()
    elif str(name).upper() == 'HEIGHT_COURSE':
        spec = read_height_spec()
    elif str(name).upper() == 'TURN_ENVELOPE':
        spec = read_turn_spec()
    return selection.apply_policy_experiment(
        env_cfg, name, train_cfg, spec=spec,
        stand_randomization_level=os.environ.get('FUDAN_STAND_RANDOMIZATION_LEVEL'),
    )


def apply_training_profile(env_cfg, train_cfg=None, *, profile='method_v1',
                           phase='stand', level=None):
    return primitives.apply_training_profile(
        env_cfg, train_cfg, profile=profile, phase=phase, level=level,
        stand_randomization_level=os.environ.get('FUDAN_STAND_RANDOMIZATION_LEVEL'),
    )


def apply_stand_balance(env_cfg, train_cfg, name):
    from wheel_legged_gym.experiments.recipes.stand_balance import apply_stand_balance as apply
    return apply(env_cfg, train_cfg, name,
                 stand_randomization_level=os.environ.get('FUDAN_STAND_RANDOMIZATION_LEVEL'))


def validate_motion_source(path, resume, mode, cfg):
    return recipe_source.validate_motion_source(
        path, resume, mode, cfg, spec=read_motion_spec(),
    )


def validate_height_source(path, resume, mode, cfg):
    return recipe_source.validate_height_source(
        path, resume, mode, cfg, spec=read_height_spec(),
    )


def validate_turn_source(path, resume, mode, cfg):
    return recipe_source.validate_turn_source(
        path, resume, mode, cfg, spec=read_turn_spec(),
    )
