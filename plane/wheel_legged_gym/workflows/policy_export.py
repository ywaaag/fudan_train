"""Export and numerically verify a checkpoint before exposing the output path."""
from pathlib import Path
from wheel_legged_gym.ports.processes import PythonJob


def export_verified_policy(checkpoint: Path, tag: str, *, plane: Path,
                           job: Path, run: PythonJob) -> Path:
    onnx = job / (tag + '.onnx')
    run([plane/'export_onnx/export_onnx.py', '--log_root='+str(plane/'logs/wheel_legged'),
         '--load_run='+checkpoint.parent.name, '--checkpoint='+checkpoint.stem.split('_')[-1],
         '--out='+str(onnx)], tag+'_export')
    run([plane/'export_onnx/verify_onnx.py', '--checkpoint='+str(checkpoint),
         '--onnx='+str(onnx), '--batch=256'], tag+'_onnx_check')
    return onnx
