# 仿真张量的所有权

`adapters/isaacgym/state_tensors.acquire_state_tensors`负责按原顺序获取/刷新Gym张量，
再构造SimulationTensors。环境显式接收8个字段，不用动态setattr或隐式代理。

| 字段 | 存储所有权 |
|---|---|
| root_states、dof_state | gymtorch.wrap_tensor包装仿真存储 |
| dof_pos、dof_vel、base_quat | 上述张量的共享视图 |
| contact_forces、rigid_body_states | 仿真存储的reshape视图 |
| dof_acc | 独立zeros_like分配 |

冻结dataclass只禁止替换字段，不禁止写入张量。共享视图必须保留：复制dof_pos会导致
reset/控制读取与仿真状态脱节。不得把copy/clone作为表面上的“隔离优化”。

test_state_tensors.py用可记录调用的Gym替身检查获取/刷新顺序，并写入每类视图确认
原存储同步变化、dof_acc独立。该测试不替代真实GPU生命周期验证。
后续历史buffer、随机化buffer及课程统计仍由环境持有，尚未全部拆分。

## 命令指标存储

`domain/commands/metrics_state.py`提供CommandMetrics和create_command_metrics。
25个累计字段均有明确名字，环境只持有command_metrics，不保留command_metric_*别名。
评估脚本也通过env.command_metrics读取指标，不用setattr隐式新增属性。
buffers()按原顺序返回同一批张量，环境reset时对该列表清零仍会同步修改具名字段。
冻结记录只冻结绑定，张量由任务实例独立拥有；工厂不改变RNG。

test_command_metrics_state.py检查字段/列表别名、不同字段/实例独立、dtype/shape以及RNG。
历史/随机化buffer仍需继续整理。

`domain/commands/tracking_metrics.accumulate`现负责每步命令指标累计，输入显式列出命令、
速度、torque及阈值，输出是原地更新CommandMetrics。zero_mask与method_wheel_terms是
计算回调，保留原调用时机；它们只应计算当前状态，不执行仿真或改变课程。
环境保留method/legacy在命令重采样前后累计的原时序。legacy slip仍是平方误差，
method slip仍取residual_rms，不能因字段名相似就擅自统一公式。

test_tracking_metrics_equivalence.py对两种模式各累计三次，全部25字段及回调顺序
与冻结旧实现一致。其他历史/随机化buffer仍待整理。

`tracking_metrics.summarize(metrics, env_ids)`在reset前汇总选中环境，分别以总步数、
静止步数、前进步数和后退步数归一化，分母仍clamp到至少1。返回日志字段而不修改buffer。
环境保持“汇总→课程累计/更新→物理reset→统计清零”的顺序。测试覆盖零计数和非零计数，
与原reset片段的18个输出字段逐项精确对照。

## 物理reset

`adapters/isaacgym/state_reset`公开reset_dofs/reset_root_states，显式接收共享状态张量
及Gym句柄。只改选中的env行，用int32索引提交给Gym；episode统计/课程更新不在此模块。
DOF重置使用已有default_dof_pos，不额外随机缩放；原注释对此不准确，已纠正说明。
root在custom_origins时先XY扰动，再抽取六通道速度；普通origin只抽速度。
test_state_reset.py验证选中/未选中行、共享存储提交、索引dtype和RNG状态。
