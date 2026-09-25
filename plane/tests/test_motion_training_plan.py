"""Compare manifests with frozen supervisor logic, including omitted/zero values."""
import importlib.util
import json
from pathlib import Path

import pytest

from wheel_legged_gym.workflows.motion_training_plan import (
    TrainingOptions, experiment_spec, experiment_signature, training_arguments,
)

reference_path = Path(__file__).parent/'fixtures/motion_training_plan_reference.py'
reference_spec = importlib.util.spec_from_file_location('motion_plan_reference', reference_path)
reference = importlib.util.module_from_spec(reference_spec)
reference_spec.loader.exec_module(reference)


@pytest.mark.parametrize('ramp', [None, .5, 2.])
@pytest.mark.parametrize('fraction', [0., .25, .5])
@pytest.mark.parametrize('mode', ['default', 'geometry', 'recovery', 'dynamic'])
def test_manifest_and_duplicate_signature_match_original(ramp, fraction, mode):
    options = TrainingOptions(
        start_stop_ramp_seconds=ramp, start_stop_fraction=fraction,
        geometry_symmetry=mode=='geometry', recover_motion=mode=='recovery',
        command_switch_seconds=5 if mode=='dynamic' else 0,
        freeze_motion_encoder=mode=='dynamic',
        geometry_weight=-.2 if mode=='dynamic' else -.1,
        dynamic_fixed_lr=1e-6 if mode=='dynamic' else None,
        dynamic_equivariance=.01 if mode=='dynamic' else None,
        dynamic_reference_coef=1. if mode=='dynamic' else None,
    )
    source = Path('/unused/run/model_10200.pt')
    focus = ['forward4', 'backward4', 'forward4']
    old_spec, old_signature = reference.original_spec(
        source, 'basic_motion', focus, options, lambda _: 'source-sha')
    new_spec = experiment_spec(source, 'source-sha', 'basic_motion', focus, options)
    assert json.dumps(new_spec, indent=2)==json.dumps(old_spec, indent=2)
    assert experiment_signature(new_spec, options.training_seed)==old_signature
    # A plan owns its focus list, so later recovery changes cannot rewrite it.
    focus.clear()
    assert new_spec['focus']==['forward4', 'backward4', 'forward4']


def test_resume_command_and_seed_identity():
    source = Path('/unused/run/model_10200.pt')
    assert training_arguments(Path('/plane'), source, 19)==[
        Path('/plane/wheel_legged_gym/scripts/train.py'), '--task=wheel_legged',
        '--headless', '--resume', '--resume_mode=full', '--load_run=run',
        '--checkpoint=10200', '--policy_experiment=MOTION_GOAL', '--seed=19',
    ]
    spec = experiment_spec(source, 'sha', 'basic_motion', [], TrainingOptions())
    assert experiment_signature(spec, 19)!=experiment_signature(spec, 23)
