"""Exercise actual environment delegates against frozen curriculum control flow."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheel_legged_gym.domain.commands.command_curriculum import evaluate_curriculum_window
from wheel_legged_gym.domain.commands.staged_progress import window_is_due, advance_stage, log_metrics


def implementation_types():
    root = Path(__file__).parent
    spec = importlib.util.spec_from_file_location('original_progress', root / 'fixtures/staged_progress_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse((root.parent / 'wheel_legged_gym/envs/base/legged_robot.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LeggedRobot')
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ('_maybe_advance_staged_curriculum', '_staged_curriculum_log_metrics')]
    scope = dict(evaluate_curriculum_window=evaluate_curriculum_window,
                 window_is_due=window_is_due, advance_stage=advance_stage, staged_log_metrics=log_metrics)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'environment_progress', 'exec'), scope)
    return module.OriginalProgress, scope['LeggedRobot']


@pytest.mark.parametrize('stage,streak,passed,step,episodes', [
    (0, 0, True, 10, 2), (0, 1, True, 10, 2), (0, 1, False, 10, 2),
    (1, 1, True, 10, 2), (0, 1, True, 9, 2), (0, 1, True, 10, 1),
])
def test_progress_and_callback_order_match(stage, streak, passed, step, episodes):
    results = []
    for cls in implementation_types():
        obj = cls()
        obj.cfg = NS(commands=NS(curriculum_check_interval_steps=10, curriculum_min_episodes=2,
            curriculum_required_passes=2, curriculum_min_survival=.9, curriculum_max_zero_vx=.05,
            curriculum_max_zero_yaw=.1, curriculum_max_linear_relative_error=.1,
            curriculum_max_yaw_relative_error=.1))
        obj.common_step_counter = step
        obj.command_curriculum_last_check_step = 0
        obj.command_curriculum_window_episodes = episodes
        obj.command_curriculum_stages = ((.5, .5), (1., 1.))
        obj.command_curriculum_stage = stage
        obj.command_curriculum_pass_streak = streak
        obj.command_curriculum_last_metrics = {'passed': False, 'enough_samples': False}
        obj.command_curriculum_window = {k: torch.tensor(float(v)) for k, v in dict(
            timeouts=episodes if passed else 0, zero_count=1, zero_abs_vx_sum=0,
            zero_abs_yaw_sum=0, reverse_command_abs_sum=1, reverse_error_sum=0,
            forward_command_abs_sum=1, forward_error_sum=0, yaw_command_abs_sum=1,
            yaw_error_sum=0).items()}
        calls = []
        obj._uses_staged_curriculum = lambda: True
        obj._apply_staged_curriculum_ranges = lambda: calls.append(('ranges', obj.command_curriculum_stage))
        obj._reset_staged_curriculum_window = lambda: calls.append(('reset', obj.command_curriculum_stage))
        obj._maybe_advance_staged_curriculum()
        results.append((obj.command_curriculum_stage, obj.command_curriculum_pass_streak,
                        obj.command_curriculum_last_check_step, copy.deepcopy(calls),
                        obj._staged_curriculum_log_metrics()))
    assert results[0] == results[1]
