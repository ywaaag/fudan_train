"""First yaw curriculum with explicit retention of every accepted speed2 anchor."""
import hashlib
import json
from pathlib import Path
from .legacy_speed2_stop import apply_legacy_speed2_stop


def apply_legacy_yaw(cfg, train):
    manifest = apply_legacy_speed2_stop(cfg, train)
    bank = [(0., 0.)] * 8 + [(v, 0.) for v in (-.5, .5, -1., 1., -1.5, 1.5, -2., 2.)]
    bank += [(0., -.5), (0., .5)] * 2
    cfg.commands.sampling_strategy = 'fixed_bank'
    cfg.commands.fixed_bank = bank
    cfg.commands.ranges.ang_vel_yaw = [-.5, .5]
    cfg.commands.training_profile = 'legacy_yaw05_v1'
    cfg.commands.training_phase = 'yaw'
    manifest.pop('exact_command_fractions', None)
    manifest.update(name='LEGACY_YAW', profile='legacy_yaw05_v1',
        command_bank=bank, command_semantics='20 equal slots held until episode reset',
        ablation_variable='sampling only: zero40%, eight translation anchors40%, pure yaw +/-0.5 20%',
        scope='Pure yaw introduction; combined turns and within-episode switches are evaluation-only')
    return manifest


def validate_source(path, resume, mode, cfg):
    if not resume or mode != 'full':
        raise ValueError('Yaw curriculum requires full resume')
    path = Path(path).resolve()
    record = json.loads((Path(__file__).resolve().parents[4] / 'docs/data/legacy_speed2_passed_20260920.json').read_text())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != record['sha256'] or record['passed'] != 27:
        raise ValueError('Expected independently accepted speed2 model1400')
    old = json.loads((path.parent / 'policy_experiment.json').read_text())
    if old['profile'] != 'legacy_speed2_stop_v2':
        raise ValueError('Unexpected source profile')
    for key, value in old['reward_scales'].items():
        if getattr(cfg.rewards.scales, key, 0.) != value:
            raise ValueError('Reward mismatch: ' + key)
    return {'source_checkpoint': str(path), 'source_checkpoint_sha256': sha}
