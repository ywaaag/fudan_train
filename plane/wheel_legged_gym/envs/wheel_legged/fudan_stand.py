"""Fudan reference reward/PPO reproduction for the custom tree standing task."""

FUDAN_SCALES = {
    'tracking_lin_vel': 1., 'tracking_lin_vel_enhance': 1.,
    'tracking_ang_vel': 1., 'tracking_ang_vel_enhance': 1.,
    'base_height': 1., 'nominal_state': -1., 'lin_vel_z': -1.,
    'ang_vel_xy': -.20, 'orientation': -100., 'dof_vel': -5e-5,
    'dof_acc': -2.5e-7, 'torques': -.0001, 'action_rate': -.01,
    'action_smooth': -.01, 'collision': -1., 'dof_pos_limits': -1.,
}
# Overrides from wheel_leg_mjrl-lqr/fudan_train: the user's adapted reference.
FUDAN_SCALES.update(orientation=-500., base_height=2., dof_vel=-.01,
                    tracking_lin_vel_enhance=0., tracking_ang_vel_enhance=0.,
                    zero_base_velocity=-1., zero_wheel_velocity=-1.)


def apply_fudan_stand(env_cfg, train_cfg):
    from .policy_experiments import _apply_method_randomization
    if train_cfg is None:
        raise ValueError('FUDAN_STAND requires train_cfg')
    env_cfg.commands.training_profile = 'fudan_stand_v1'
    env_cfg.commands.training_phase = 'stand'
    env_cfg.commands.sampling_strategy = 'uniform'
    env_cfg.commands.curriculum = False
    env_cfg.commands.curriculum_mode = 'disabled'
    env_cfg.commands.curriculum_stages = ()
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [0., 0.]
    env_cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    env_cfg.commands.ranges.height = [.40, .40]
    env_cfg.commands.mixture_zero_fraction = 1.
    env_cfg.commands.mixture_small_fraction = 0.
    env_cfg.commands.mixture_reverse_fraction = 0.
    env_cfg.commands.mixture_forward_fraction = 0.
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.rewards.reward_pipeline = 'legacy_v0'
    env_cfg.rewards.unclipped_reward_names = ()
    env_cfg.rewards.only_positive_rewards = False
    env_cfg.rewards.clip_single_reward = 1.
    env_cfg.rewards.tracking_sigma = .25
    env_cfg.rewards.soft_dof_pos_limit = .97
    for name in dir(env_cfg.rewards.scales):
        if not name.startswith('_') and isinstance(getattr(env_cfg.rewards.scales, name), (int, float)):
            setattr(env_cfg.rewards.scales, name, 0.)
    for name, value in FUDAN_SCALES.items():
        setattr(env_cfg.rewards.scales, name, value)
    # Reference lists are empty; retain this for the reproduction baseline.
    # Evaluate forbidden contact independently rather than claim safety here.
    env_cfg.asset.penalize_contacts_on = []
    env_cfg.asset.terminate_after_contacts_on = []
    env_cfg.domain_rand_level = 0
    _apply_method_randomization(env_cfg, 0)
    for name in ('randomize_restitution', 'randomize_default_dof_pos',
                 'randomize_action_delay', 'lift_robots', 'downward_impulse_robots', 'vmc_force_events'):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)
    train_cfg.policy.init_noise_std = .5
    settings = dict(value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2,
                    entropy_coef=.01, num_learning_epochs=5, num_mini_batches=4,
                    learning_rate=1e-3, schedule='adaptive', gamma=.99, lam=.95,
                    desired_kl=.005, max_grad_norm=1., extra_learning_rate=1e-3,
                    symmetry_loss_coef=0.)
    for name, value in settings.items():
        setattr(train_cfg.algorithm, name, value)
    train_cfg.runner.num_steps_per_env = 48
    train_cfg.runner.save_interval = 100
    return dict(name='FUDAN_STAND', profile='fudan_stand_v1', phase='stand', level=0,
                reference='/home/kellen/wheel_leg_mjrl-lqr/fudan_train/plane/wheel_legged_gym',
                reward_version='fudan_legacy_stand_v1', reward_scales=dict(FUDAN_SCALES),
                command_limits=[0., 0.],
                adaptations=['custom tree asset/PD unchanged', 'height 0.40',
                             'zero vx/yaw', 'root-frame hip-to-wheel virtual angle',
                             'initial no-randomization baseline'],
                optimizer={k:settings[k] for k in ('learning_rate','extra_learning_rate','schedule','entropy_coef')})
