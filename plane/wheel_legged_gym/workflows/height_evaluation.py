"""Evaluate a height checkpoint on the supplied full command bank."""
from wheel_legged_gym.evaluation.height_acceptance import assess_height_row
from wheel_legged_gym.ports.artifacts import EvaluationArtifacts
from wheel_legged_gym.ports.processes import PythonJob


def evaluate_height_checkpoint(candidate, stage, bank, *, evaluation_script,
                               files: EvaluationArtifacts, run: PythonJob):
    """Retain all three seeds even after failed rows; exceptions stop immediately."""
    records = []
    for seed in [19, 37, 53]:
        tag = f'{stage}_seed{seed}'
        out = files.directory / (tag + '.json')
        run([evaluation_script, '--checkpoint=' + str(candidate),
             '--out=' + str(out), '--seed=' + str(seed), '--commands']
            + [str(v) for v, w, h in bank]
            + ['--yaw-commands'] + [str(w) for v, w, h in bank]
            + ['--height-commands'] + [str(h) for v, w, h in bank], tag)
        data = files.read_json(out)
        for row in data['results']:
            assess_height_row(row, stage)
            records.append(dict(row, seed=seed))
    result = {
        'checkpoint': str(candidate), 'stage': stage,
        'passed': all(r['gate']['passed'] for r in records),
        'passed_count': sum(r['gate']['passed'] for r in records),
        'total': len(records), 'records': records,
    }
    files.write_json(stage + '_acceptance.json', result)
    return result
