"""Asset options and Gym query ordering remain explicit at the simulator boundary."""
from types import SimpleNamespace as NS

from wheel_legged_gym.adapters.isaacgym.robot_asset import load_robot_asset


def test_load_options_query_order_and_property_identity():
    fields = dict(default_dof_drive_mode=3, collapse_fixed_joints=True,
        replace_cylinder_with_capsule=False, flip_visual_attachments=True,
        fix_base_link=False, density=42., angular_damping=.1, linear_damping=.2,
        max_angular_velocity=50., max_linear_velocity=20., armature=.03,
        thickness=.005, disable_gravity=False)
    cfg = NS(file='{WHEEL_LEGGED_GYM_ROOT_DIR}/assets/robot.urdf', **fields)
    calls = []
    properties, shapes = object(), object()
    class Gym:
        def load_asset(self, sim, root, file, options):
            assert (sim, root, file) == ('sim', '/package/assets', 'robot.urdf')
            for key, value in fields.items():
                actual = getattr(options, key)
                if isinstance(value, float):
                    assert abs(actual - value) < 1e-6
                else:
                    assert actual == value
            calls.append('load')
            return 'asset'
        def get_asset_dof_count(self, asset):
            calls.append('dof_count'); return 6
        def get_asset_rigid_body_count(self, asset):
            calls.append('body_count'); return 99
        def get_asset_dof_properties(self, asset):
            calls.append('dof_properties'); return properties
        def get_asset_rigid_shape_properties(self, asset):
            calls.append('shape_properties'); return shapes
        def get_asset_rigid_body_names(self, asset):
            calls.append('body_names'); return ['base', 'wheel']
        def get_asset_dof_names(self, asset):
            calls.append('dof_names'); return ['dof'] * 6
    result = load_robot_asset(Gym(), 'sim', cfg, '/package')
    assert calls == ['load', 'dof_count', 'body_count', 'dof_properties',
                     'shape_properties', 'body_names', 'dof_names']
    assert result.num_bodies == 2  # Original code replaces queried count with len(names).
    assert result.num_dof == result.num_dofs == 6
    assert result.dof_properties is properties
    assert result.rigid_shape_properties is shapes
