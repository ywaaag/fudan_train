"""Shared evaluation services; never imports a command-line entry."""
from types import SimpleNamespace
from isaacgym import gymapi

def build_evaluation_args() -> SimpleNamespace:
    return SimpleNamespace(
        task="wheel_legged",
        sim_device="cuda:0",
        sim_device_type="cuda",
        sim_device_id=0,
        compute_device_id=0,
        graphics_device_id=-1,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=True,
        use_gpu_pipeline=True,
        pipeline="gpu",
        subscenes=0,
        num_threads=0,
        headless=True,
        rl_device="cuda:0",
        num_envs=1,
        seed=1,
        max_iterations=None,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        horovod=False,
        exptid="",
    )

def disable_evaluation_randomization(cfg) -> None:
    cfg.noise.add_noise = False
    names = (
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_mass",
        "randomize_inertia",
        "randomize_base_com",
        "randomize_Kp",
        "randomize_Kd",
        "randomize_motor_torque",
        "randomize_default_dof_pos",
        "randomize_action_delay",
        "push_robots",
        "lift_robots",
        "downward_impulse_robots",
        "vmc_force_events",
    )
    for name in names:
        if hasattr(cfg.domain_rand, name):
            setattr(cfg.domain_rand, name, False)

