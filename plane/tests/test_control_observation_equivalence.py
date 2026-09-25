"""Compare actual environment delegates with frozen pre-refactor behavior."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheel_legged_gym.domain.control.actuation import mixed_pd_torques
from wheel_legged_gym.domain.observations.critic import privileged_observation
from wheel_legged_gym.domain.observations.policy import (
    noise_scale_vector, proprioception, update_history,
)


def implementations():
    """Compile just these environment methods; CPU tests need no Isaac runtime."""
    root = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "frozen_control", root / "fixtures/control_observation_reference.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    original = reference.OriginalControlObservations
    names = {name for name in original.__dict__ if callable(getattr(original, name))}
    tree = ast.parse((root.parent / "wheel_legged_gym/envs/base/legged_robot.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot")
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(torch=torch, mixed_pd_torques=mixed_pd_torques,
                 privileged_observation=privileged_observation,
                 noise_scale_vector=noise_scale_vector, proprioception=proprioception,
                 update_history=update_history)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "environment_delegates", "exec"), scope)
    return original, scope["LeggedRobot"]


@pytest.mark.parametrize("seed", [11, 43])
@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("privileged", [False, True])
def test_control_observation_and_rng_are_exact(seed, method, noise, privileged):
    torch.manual_seed(seed)
    original_type, migrated_type = implementations()
    before = original_type()
    n = 4
    before.num_envs = n
    before.num_obs = 25
    before.cfg = NS(
        control=NS(pos_action_scale=0.5, vel_action_scale=10.0, decimation=2),
        env=NS(num_privileged_obs=64 if privileged else None, obs_history_dec=2),
        noise=NS(add_noise=noise, noise_level=0.7, noise_scales=NS(
            ang_vel=0.1, gravity=0.02, dof_pos=0.03, dof_vel=0.04,
            height_measurements=0.05)),
        terrain=NS(measure_heights=True),
    )
    before.obs_scales = NS(ang_vel=0.25, dof_pos=1.0, dof_vel=0.05,
                          height_measurements=5.0, lin_vel=2.0, dof_acc=0.1, torque=0.01)
    shapes = {
        "base_ang_vel": (n, 3), "projected_gravity": (n, 3), "commands": (n, 3),
        "commands_scale": (3,), "dof_pos": (n, 6), "default_dof_pos": (n, 6),
        "dof_vel": (n, 6), "actions": (n, 6), "p_gains": (n, 6),
        "d_gains": (n, 6), "torques_scale": (n, 6), "torque_limits": (6,),
        "command_metric_preclip_torque_saturation_sum": (n,), "root_states": (n, 13),
        "measured_heights": (n, 7), "base_lin_vel": (n, 3), "last_actions": (n, 6, 2),
        "dof_acc": (n, 6), "torques": (n, 6), "base_mass": (n,), "base_com": (n, 3),
        "raw_default_dof_pos": (n, 6), "friction_coef": (n,), "restitution_coef": (n,),
        "obs_buf": (n, 25), "obs_history": (n, 125),
    }
    for name, shape in shapes.items():
        setattr(before, name, torch.randn(shape))
    before.torque_limits = before.torque_limits.abs() + 0.1
    before._method_v1 = method
    before.envs_steps_buf = torch.tensor([0, 1, 4, 6])
    after = migrated_type()
    after.__dict__.update(copy.deepcopy(before.__dict__))
    after.command_metrics = NS(preclip_torque_saturation_sum=
        after.__dict__.pop('command_metric_preclip_torque_saturation_sum'))
    rng = torch.get_rng_state()
    for obj in (before, after):
        torch.set_rng_state(rng)
        obj.noise_scale_vec = obj._get_noise_scale_vec(obj.cfg)
        obj.torques = obj._compute_torques(obj.actions)
        obj.compute_observations()
        obj.rng_after = torch.get_rng_state()
    for name, value in before.__dict__.items():
        if torch.is_tensor(value):
            actual = (after.command_metrics.preclip_torque_saturation_sum
                      if name=='command_metric_preclip_torque_saturation_sum' else getattr(after, name))
            assert torch.equal(value, actual), name
    assert before.add_noise == after.add_noise
