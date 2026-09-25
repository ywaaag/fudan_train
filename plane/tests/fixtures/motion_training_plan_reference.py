"""Frozen pre-extraction manifest logic, 2026-09-24; do not update from replacement."""
import json

def original_spec(source, stage, focus, opts, digest):
    iteration=int(source.stem.split('_')[-1])
    spec={'stage':stage,'focus':focus,'source_checkpoint':str(source),
          'source_sha256':digest(source),'source_iteration':iteration}
    if opts.geometry_symmetry or opts.recover_motion:spec['geometry_symmetry']=True
    if opts.command_switch_seconds:spec['command_switch_seconds']=opts.command_switch_seconds
    if opts.freeze_motion_encoder:spec['freeze_motion_encoder']=True
    if opts.geometry_weight != -.1:spec['geometry_weight']=opts.geometry_weight
    if opts.start_stop_ramp_seconds:
        spec['start_stop_ramp_seconds']=opts.start_stop_ramp_seconds
        spec['start_stop_speed']=opts.start_stop_speed
        if opts.start_stop_fraction!=.5:spec['start_stop_fraction']=opts.start_stop_fraction
        if opts.dynamic_fixed_lr:spec['dynamic_fixed_lr']=opts.dynamic_fixed_lr
        if opts.dynamic_equivariance:spec['dynamic_equivariance']=opts.dynamic_equivariance
        if opts.dynamic_reference_coef:spec['dynamic_reference_coef']=opts.dynamic_reference_coef
    signature=json.dumps({'spec':spec,'seed':opts.training_seed},sort_keys=True)
    return spec, signature
