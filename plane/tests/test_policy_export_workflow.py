"""Export validation is ordered, mandatory and failure-propagating."""
from pathlib import Path
import pytest
from wheel_legged_gym.workflows.policy_export import export_verified_policy


def test_export_then_verification_preserves_cli_arguments():
    calls=[]
    checkpoint=Path('/plane/logs/wheel_legged/run/model_10200.pt')
    result=export_verified_policy(checkpoint,'accepted',plane=Path('/plane'),job=Path('/job'),
                                  run=lambda args,tag:calls.append((args,tag)))
    assert calls==[
        ([Path('/plane/export_onnx/export_onnx.py'),'--log_root=/plane/logs/wheel_legged',
          '--load_run=run','--checkpoint=10200','--out=/job/accepted.onnx'],'accepted_export'),
        ([Path('/plane/export_onnx/verify_onnx.py'),'--checkpoint='+str(checkpoint),
          '--onnx=/job/accepted.onnx','--batch=256'],'accepted_onnx_check'),
    ]
    assert result==Path('/job/accepted.onnx')


@pytest.mark.parametrize('failure_at', [1,2])
def test_failure_does_not_return_an_accepted_artifact(failure_at):
    calls=[]
    def run(args,tag):
        calls.append(tag)
        if len(calls)==failure_at:raise RuntimeError('failed job')
    with pytest.raises(RuntimeError,match='failed job'):
        export_verified_policy(Path('/run/model_1.pt'),'candidate',plane=Path('/plane'),job=Path('/job'),run=run)
    assert len(calls)==failure_at
