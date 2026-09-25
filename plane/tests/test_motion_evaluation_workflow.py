"""Workflow uses explicit ports and validates cached evaluation provenance."""
import json

import pytest

from wheel_legged_gym.adapters.artifacts.job_files import JobFiles
from wheel_legged_gym.workflows.motion_evaluation import EvaluationOptions, evaluate


def record():
    return {'checkpoint_sha256': 'expected', 'results': [{
        'command': [0., 0., .4], 'failure_count': 0, 'timeout_count': 0,
        'nonwheel_contact_full_fraction': 0.,
        'metrics': dict(vx_mae=0., yaw_mae=0., abs_yaw=0., nonwheel_contact=0.,
                        left_contact=1., right_contact=1., height_mae=0., torque_saturation=0.,
                        knee_mirror_m=.02, wheel_mirror_m=.02),
    }]}


@pytest.mark.parametrize('cached', [False, True])
def test_evaluate_runs_only_missing_results_and_preserves_geometry(tmp_path, cached):
    files = JobFiles(tmp_path)
    if cached:
        files.write_json('case_seed19.json', record())
    calls = []
    def run(arguments, tag):
        calls.append((arguments, tag))
        output = next(arg.split('=', 1)[1] for arg in arguments if str(arg).startswith('--out='))
        from pathlib import Path
        Path(output).write_text(json.dumps(record()))
    result = evaluate(tmp_path/'model.pt', [(0., 0.)], [19], 'case',
        options=EvaluationOptions(None, 1., True, True, .03), recovery_geometry={'0.0': .02},
        evaluation_script='evaluator.py', files=files, run=run, digest=lambda _: 'expected')
    assert len(calls) == (0 if cached else 1)
    assert result['passed'] and result['posture_retained']
    assert result['geometry_by_command'] == {'0.0': .02}
    assert files.read_json(tmp_path/'case_summary.json')['passed']


def test_cached_result_with_wrong_checkpoint_is_rejected(tmp_path):
    files = JobFiles(tmp_path)
    files.write_json('case_seed19.json', record())
    with pytest.raises(RuntimeError, match='Evaluation source changed'):
        evaluate(tmp_path/'model.pt', [(0., 0.)], [19], 'case',
            options=EvaluationOptions(None, 1., False, False, .03), recovery_geometry={},
            evaluation_script='unused', files=files, run=lambda *args: pytest.fail('cached'),
            digest=lambda _: 'different checkpoint')
    assert not (tmp_path/'case_summary.json').exists()
