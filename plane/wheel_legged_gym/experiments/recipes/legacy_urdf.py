"""Original Fudan reward/PPO semantics on the current wheel-leg URDF only.

The asset and feasible reset pose remain the port's own values. This profile is
therefore a training-config ablation, not a physics-equivalent reproduction of
the upstream infantry asset.
"""


def apply_legacy_urdf(cfg, train):
    # Original command and reward behavior. Keep the current tree asset and
    # its z=0.40/zero-joint feasible reset from wheel_legged_config.py.
    cfg.commands.curriculum = True
    cfg.commands.curriculum_mode = 'legacy'
    cfg.commands.sampling_strategy = 'uniform'
    cfg.commands.hold_command_until_reset = False
    cfg.commands.training_profile = 'legacy_urdf_v1'
    cfg.commands.training_phase = 'legacy'
    cfg.commands.ranges.lin_vel_x = [-2.0, 2.0]
    cfg.commands.ranges.ang_vel_yaw = [-2.0, 2.0]
    cfg.commands.ranges.height = [.40, .40]
    cfg.rewards.reward_pipeline = 'legacy_v0'
    cfg.rewards.only_positive_rewards = False
    cfg.rewards.tracking_sigma = .25
    scales = cfg.rewards.scales
    values = {
        'tracking_lin_vel': 1.0, 'tracking_lin_vel_enhance': 1.0,
        'tracking_ang_vel': 1.0, 'tracking_ang_vel_enhance': 1.0,
        'base_height': 1.0, 'nominal_state': -1.0,
        'lin_vel_z': -1.0, 'ang_vel_xy': -.20, 'orientation': -100.0,
        'dof_vel': -5e-5, 'dof_acc': -2.5e-7, 'torques': -.0001,
        'action_rate': -.01, 'action_smooth': -.01,
        'collision': -1.0, 'dof_pos_limits': -1.0,
        'track_vx_coarse': 0.0, 'track_vx_fine': 0.0, 'track_vx_gap': 0.0,
        'track_yaw_coarse': 0.0, 'track_yaw_fine': 0.0, 'track_yaw_gap': 0.0,
        'height_cost': 0.0, 'lateral_velocity': 0.0, 'wheel_slip': 0.0,
        'airborne_wheel_spin': 0.0, 'wheel_contact_loss': 0.0,
        'forbidden_contact': 0.0, 'torque_cost': 0.0, 'power_cost': 0.0,
        'action_second_diff': 0.0, 'method_termination': 0.0,
        'stand_bilateral_geometry': 0.0,
    }
    for name, value in values.items():
        if hasattr(scales, name): setattr(scales, name, value)
    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [.6, 1.4]
    cfg.domain_rand.randomize_restitution = True
    cfg.domain_rand.restitution_range = [.6, 1.0]
    cfg.domain_rand.randomize_base_mass = True
    cfg.domain_rand.added_mass_range = [-1.0, 2.0]
    cfg.domain_rand.randomize_inertia = True
    cfg.domain_rand.randomize_inertia_range = [.9, 1.1]
    cfg.domain_rand.randomize_base_com = True
    cfg.domain_rand.rand_com_vec = [.02, .02, .02]
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_Kp = True
    cfg.domain_rand.randomize_Kp_range = [.95, 1.05]
    cfg.domain_rand.randomize_Kd = True
    cfg.domain_rand.randomize_Kd_range = [.95, 1.05]
    cfg.domain_rand.randomize_motor_torque = True
    cfg.domain_rand.randomize_motor_torque_range = [.95, 1.05]
    cfg.domain_rand.randomize_default_dof_pos = True
    cfg.domain_rand.randomize_default_dof_pos_range = [-.03, .03]
    # Original wheel damping was 0.2; retain the current asset's explicit
    # wheel damping for the first safe comparison and record the difference.
    train.algorithm.learning_rate = 1e-3
    train.algorithm.extra_learning_rate = 1e-3
    train.algorithm.schedule = 'adaptive'
    train.algorithm.entropy_coef = .01
    train.algorithm.desired_kl = .005
    train.algorithm.num_learning_epochs = 5
    train.algorithm.num_mini_batches = 4
    train.algorithm.symmetry_loss_coef = 0.0
    manifest = {
        'name': 'LEGACY_URDF', 'profile': 'legacy_urdf_v1',
        'reward_pipeline': 'legacy_v0', 'asset_scope': 'current wheel_leg_train.urdf',
        'physics_equivalence': False,
        'upstream_config_source': '/home/kellen/fudan_rl_wheel_leg/plane',
        'reset_adaptation': 'current z=0.40 and zero joints retained for tree URDF feasibility',
        'wheel_damping_adaptation': 'current wheel damping retained at 1.0; upstream was 0.2',
        'optimizer': {'learning_rate': 1e-3, 'extra_learning_rate': 1e-3,
                     'schedule': 'adaptive', 'entropy_coef': .01},
        'command_semantics': 'upstream uniform/resampling; current wheel-leg height fixed .40',
    }
    return manifest
