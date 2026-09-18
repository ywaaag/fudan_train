"""Print compact TensorBoard tail statistics without dumping training stdout."""
import argparse
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser(__doc__)
p.add_argument('run', type=Path)
p.add_argument('--window', type=int, default=50)
a = p.parse_args()
events = EventAccumulator(str(a.run), size_guidance={'scalars': 0})
events.Reload()
keys = ['Train/mean_reward', 'Train/mean_episode_length', 'Loss/policy_symmetry',
        'Episode/rew_stand_bilateral_geometry',
        'Episode/zero_abs_vx', 'Episode/zero_abs_yaw_rate',
        'Episode/zero_abs_wheel_speed', 'Episode/wheel_contact_fraction',
        'Episode/preclip_torque_saturation_fraction']
result = {'run': str(a.run), 'metrics': {}}
for key in keys:
    if key in events.Tags()['scalars']:
        values = events.Scalars(key)[-a.window:]
        result['metrics'][key] = {'step': values[-1].step,
            'last': round(values[-1].value, 6),
            'tail_mean': round(sum(x.value for x in values)/len(values), 6)}
print(json.dumps(result, indent=2))
