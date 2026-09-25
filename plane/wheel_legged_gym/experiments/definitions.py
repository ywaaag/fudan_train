"""Read-only experiment definitions; callers receive independent manifest dictionaries."""
from collections.abc import Mapping
from types import MappingProxyType


def _readonly(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _readonly(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_readonly(item) for item in value)
    return value


def copy_definition(value):
    """Preserve tuple/scalar values while detaching every nested mapping."""
    if isinstance(value, Mapping):
        return {key: copy_definition(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(copy_definition(item) for item in value)
    return value


EXPERIMENTS = _readonly({
    # Retained historical baseline for reproducible recovery from model_15800.
    "H3": {
        "description": "validated low-speed recovery baseline",
        "zero_base_velocity_scale": -8.0,
        "zero_wheel_velocity_scale": -5.0,
        "low_speed_tracking_scale": -2.0,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.005,
        "fractions": (0.35, 0.25, 0.20, 0.20),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.50,
        "endpoint_linear_anchors": (-3.50, -3.25, -3.00, 3.00, 3.25, 3.50),
        "command_curriculum": True,
        "optimizer": {
            "learning_rate": 2.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.0002,
        },
        "save_interval": 100,
    },
    # Current focused experiment: stabilize a single 3.0 m/s operating point.
    "H7": {
        "description": "single-stage 3.0 m/s stabilization with slip and yaw shaping",
        "zero_base_velocity_scale": -5.0,
        "zero_wheel_velocity_scale": -3.0,
        "high_speed_tracking_scale": 1.0,
        "high_speed_yaw_tracking_scale": 0.0,
        "high_speed_yaw_penalty_scale": -2.0,
        "high_speed_slip_scale": -7.0,
        "unclipped_reward_names": ("high_speed_slip", "high_speed_yaw_penalty"),
        "low_speed_tracking_scale": -1.0,
        "tracking_lin_vel_scale": 1.0,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.003,
        "fractions": (0.25, 0.25, 0.25, 0.25),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.50,
        "endpoint_linear_anchors": (-3.00, 3.00),
        "command_curriculum": True,
        "curriculum_stages": ((3.0, 3.0),),
        "curriculum_initial_stage": 0,
        "curriculum_required_passes": 5,
        "optimizer": {
            "learning_rate": 3.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.0002,
        },
        "save_interval": 100,
    },
})

METHOD_V1_PHASES = _readonly({
    "stand": {
        "linear_limit": 0.0,
        "yaw_limit": 0.0,
        "randomization_level": 0,
        "fractions": (1.0, 0.0, 0.0, 0.0),
        "track_vx": (0.50, 0.25, -0.125),
        "track_yaw": (0.50, 0.25, -0.125),
        "stand_still_scale": -0.20,
    },
    "translate": {
        "linear_levels": (0.5, 1.0, 2.0, 3.0, 4.0),
        "yaw_limit": 0.0,
        "randomization_level": 0,
        "fractions": (0.20, 0.20, 0.30, 0.30),
        "track_vx": (1.00, 0.50, -0.25),
        "track_yaw": (0.50, 0.25, -0.15),
        "stand_still_scale": 0.0,
    },
    "yaw": {
        "linear_limit": 0.0,
        "yaw_levels": (0.5, 1.0, 2.0, 3.0, 4.0),
        "randomization_level": 1,
        "fractions": (0.20, 0.20, 0.30, 0.30),
        "track_vx": (0.25, 0.10, -0.10),
        "track_yaw": (1.00, 0.50, -0.25),
        "stand_still_scale": 0.0,
    },
    "combined": {
        "combined_levels": ((0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)),
        "randomization_level": 2,
        "fractions": (0.10, 0.0, 0.20, 0.70),
        "track_vx": (1.00, 0.50, -0.25),
        "track_yaw": (1.00, 0.50, -0.25),
        "stand_still_scale": 0.0,
    },
})

METHOD_V1_BASE_REWARD_SCALES = _readonly({
    "track_vx_coarse": 1.0,
    "track_vx_fine": 0.5,
    "track_vx_gap": -0.25,
    "track_yaw_coarse": 0.75,
    "track_yaw_fine": 0.35,
    "track_yaw_gap": -0.15,
    "orientation": -2.0,
    "height_cost": -1.0,
    "lin_vel_z": -0.2,
    "ang_vel_xy": -0.2,
    "lateral_velocity": -0.2,
    "wheel_slip": -1.5,
    "airborne_wheel_spin": -0.75,
    "wheel_contact_loss": -1.0,
    "forbidden_contact": -3.0,
    "torque_cost": -0.01,
    "power_cost": -0.005,
    "action_rate": -0.01,
    "action_second_diff": -0.005,
    "zero_base_velocity": -1.0,
    "zero_wheel_velocity": -1.0,
    "method_termination": -5.0,
})

