# 历史入口与兼容边界

2026-09-24用户更新：兼容旧用法不是硬要求。以下保留入口是当前迁移状态，不是永久
承诺；实际调用方迁移后优先删除冗余包装/别名，明确唯一公开API。训练/物理行为仍须等价。
MuJoCo的after_step已移除，测量只使用on_measurement快照。

旧CLI继续作为调用入口；实现迁移不允许通过静默改参数/奖励/模型契约来隐藏不兼容。
测试和真实smoke/ONNX/sim2sim证据见architecture/refactor_progress.md。

| 历史入口 | 当前实现 |
|---|---|
| rsl_rl（已移除） | learning.modules.api及learning下algorithms/runners/storage的实现模块 |
| envs/base/base_config、legged_robot_config、envs/wheel_legged/wheel_legged_config（已移除） | contracts下同名模块 |
| envs/wheel_legged实验转发（已移除） | experiments及app输入边界 |
| envs/base命令、奖励、BaseTask转发（已移除） | domain.commands、domain.rewards.terms、adapters.isaacgym.base_task |
| utils/helpers.py（已移除） | contracts.config_serialization、app.arguments/random_seed、adapters.artifacts及adapters.isaacgym责任模块 |
| utils/task_registry、utils/terrain（已移除） | app.task_registry、adapters.isaacgym.terrain_generation |
| tools/run_motion_goal.py | app.motion_supervisor |
| tools/run_candidate_closed_review.py | app.candidate_closed_review；门槛归evaluation.closed_precheck |
| tools/audit_stand_push、continue_stand_validated、run_low_speed、stand_long_guard | app同名main(root)，导入无任务副作用 |
| tools/probe_closed_initialization、probe_tree_ramp、audit_closed_mapping | adapters.mujoco进程接口 |

旧rsl_rl的21个Python转发文件已移除，学习模块只有learning命名空间。
test_learning_entrypoints.py阻止生产代码重新依赖旧路径；现有state_dict checkpoint
加载与ONNX推理已实际验证。外部Python调用方需迁移导入路径；不承诺任意历史整对象pickle
仍可反序列化。本仓库模型文件未改动，旧源码仍可通过Git/重构前备份追溯。

有意取消的隐式入口：`from wheel_legged_gym.utils import task_registry`全局实例以及
`import wheel_legged_gym.envs`自动注册。新入口为app.bootstrap.create_task_registry，
调用者创建并持有实例。这落实用户去除可变全局注册的要求；不提供隐藏懒加载单例。
旧utils包级便利导入需改为责任模块，helpers/task_registry/terrain/math已删除；logger尚保留。
quat_apply_yaw和wrap_to_pi位于domain.geometry.rotations，无调用的torch_rand_sqrt_float已删除。
现有命令行路径保留；外部Python调用方的迁移限制在此明确记录，不能称为任意导入零改动兼容。
