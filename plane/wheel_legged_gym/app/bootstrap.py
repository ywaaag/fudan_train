"""Composition root: each caller owns its task registry and configuration instances."""
def create_task_registry():
    from .task_registry import TaskRegistry
    from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
    from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
    registry = TaskRegistry()
    registry.register("wheel_legged", LeggedRobot, WheelLeggedCfg(), WheelLeggedCfgPPO())
    return registry
