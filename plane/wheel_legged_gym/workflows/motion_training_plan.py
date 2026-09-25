"""Build one motion experiment's manifest and commands without reading files."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TrainingOptions:
    training_seed: int = 23
    geometry_symmetry: bool = False
    recover_motion: bool = False
    command_switch_seconds: int = 0
    freeze_motion_encoder: bool = False
    geometry_weight: float = -.1
    start_stop_ramp_seconds: Optional[float] = None
    start_stop_speed: float = 1.
    start_stop_fraction: float = .5
    dynamic_fixed_lr: Optional[float] = None
    dynamic_equivariance: Optional[float] = None
    dynamic_reference_coef: Optional[float] = None


def experiment_spec(source: Path, source_sha256: str, stage: str, focus,
                    options: TrainingOptions):
    """Preserve omitted defaults: presence itself can select a recipe branch."""
    spec = {'stage': stage, 'focus': list(focus), 'source_checkpoint': str(source),
            'source_sha256': source_sha256,
            'source_iteration': int(source.stem.split('_')[-1])}
    if options.geometry_symmetry or options.recover_motion:
        spec['geometry_symmetry'] = True
    if options.command_switch_seconds:
        spec['command_switch_seconds'] = options.command_switch_seconds
    if options.freeze_motion_encoder:
        spec['freeze_motion_encoder'] = True
    if options.geometry_weight != -.1:
        spec['geometry_weight'] = options.geometry_weight
    if options.start_stop_ramp_seconds:
        spec['start_stop_ramp_seconds'] = options.start_stop_ramp_seconds
        spec['start_stop_speed'] = options.start_stop_speed
        if options.start_stop_fraction != .5:
            spec['start_stop_fraction'] = options.start_stop_fraction
        if options.dynamic_fixed_lr:
            spec['dynamic_fixed_lr'] = options.dynamic_fixed_lr
        if options.dynamic_equivariance:
            spec['dynamic_equivariance'] = options.dynamic_equivariance
        if options.dynamic_reference_coef:
            spec['dynamic_reference_coef'] = options.dynamic_reference_coef
    return spec


def experiment_signature(spec, training_seed: int):
    """Identify duplicate experiments using exactly the persisted spec and seed."""
    return json.dumps({'spec': spec, 'seed': training_seed}, sort_keys=True)


def training_arguments(plane: Path, source: Path, training_seed: int):
    """Common arguments; the caller appends smoke or full-round size and name."""
    iteration = int(source.stem.split('_')[-1])
    return [plane/'wheel_legged_gym/scripts/train.py', '--task=wheel_legged', '--headless',
            '--resume', '--resume_mode=full', '--load_run='+source.parent.name,
            '--checkpoint='+str(iteration), '--policy_experiment=MOTION_GOAL',
            '--seed='+str(training_seed)]
