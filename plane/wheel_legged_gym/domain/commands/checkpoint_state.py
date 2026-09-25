"""Course checkpoint payloads; no simulator, files or runner ownership."""
import torch


def method_state(commands, randomization_level, segment_counter):
    return {
        'reward_pipeline': 'normalized_v1',
        'training_profile': 'method_v1',
        'training_phase': getattr(commands, 'training_phase', 'stand'),
        'command_level': int(getattr(commands, 'method_v1_level', 0)),
        'randomization_level': int(randomization_level),
        'segment_counter': segment_counter.detach().cpu(),
    }


def restore_method_counter(state, counter, device):
    """Update counter in place only when the stored shape matches, as before."""
    if not state:
        return
    if state.get('reward_pipeline') not in {None, 'normalized_v1'}:
        raise ValueError('checkpoint reward pipeline is incompatible with method_v1')
    value = state.get('segment_counter')
    if value is not None:
        value = torch.as_tensor(value, device=device, dtype=torch.long)
        if value.shape == counter.shape:
            counter[:] = value


def staged_state(stage, pass_streak, last_metrics):
    return {'staged_command_curriculum': {
        'stage': stage, 'pass_streak': pass_streak, 'last_metrics': last_metrics,
    }}


def restored_stage(state, stages):
    """Validate stage and return its payload; caller preserves assignment order."""
    if not state:
        return None
    curriculum = state.get('staged_command_curriculum')
    if curriculum is None:
        return None
    final_stage = len(stages) - 1
    stage = int(curriculum.get('stage', 0))
    if not 0 <= stage <= final_stage:
        raise ValueError(f'invalid command curriculum stage in checkpoint: {stage}')
    return stage, curriculum
