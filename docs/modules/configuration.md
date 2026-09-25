# 配置与启动工具

配置转换入口是 `contracts/config_serialization.py`，只处理 Python 配置对象，不导入
Torch 或仿真器。字段枚举、列表递归和更新语义保持原实现。

`app/arguments.py` 解析旧训练 CLI 并覆盖配置；`app/random_seed.py` 在启动时设置随机种子。
这两个模块是进程启动边界，核心奖励、命令和配置结构不得反向导入它们。

`adapters/isaacgym/simulation_parameters.py` 将 CLI/config 转成 Isaac Gym SimParams。
`adapters/artifacts/checkpoints.py` 保留旧 checkpoint 路径选择规则；
`jit_export.py` 负责旧 TorchScript 导出，不改变 ONNX 接口。

旧 `utils/helpers.py` 已删除；调用者直接导入上面的责任模块。

`app/task_registry.py` 保存历史顶层文件，并调用
`adapters/artifacts/source_snapshot.save_source_snapshot` 增补包内 Python 源码树及 SHA256。
读取 run 时，实际公式和配方应从 `source_snapshot/` 查找，不能只看旧包装文件。
新增快照不修改策略、随机种子、训练参数或模型键；也不回填历史 run。

## 评估公开入口

`adapters/isaacgym/evaluation_setup`公开build_evaluation_args和
disable_evaluation_randomization；`policy_io`公开load_policy和set_fixed_command_ranges。
Isaac评估CLI及audit工具直接使用这些名字，不再从脚本或适配器导入私有函数。
旧_gym_args/_disable_randomization/_set_fixed_ranges别名已删除，只使用公开名字。

实验基础配置只公开apply_method_randomization，旧下划线别名已删除。
本次只改接口命名与调用路径，函数体、参数、配置修改顺序均保留。

## 配置实例所有权

BaseConfig仍支持历史嵌套类声明，但实例化时会深复制list/dict/set/tuple默认值。
因此init_state.pos、default_joint_angles和网络层列表归各实例持有，修改一个实验
不会污染类默认值或后续实验。标量默认值及配置字段名不变，旧构造入口不变。
这是移除共享可变状态的明确边界变化：不支持通过实例修改类级容器来隐式更改其他实验。
test_config_instance_isolation.py验证嵌套容器和实际robot/PPO配置隔离。

## Torch进程设置

app.torch_runtime.configure_isaac_torch_runtime显式关闭原有两个JIT profiling开关，
由TaskRegistry.make_env在解析仿真参数后、创建环境前执行，无导入副作用。
直接构造环境的外部调用者必须主动调用此入口；环境类不再隐式修改全局运行时。
这是旧Isaac/PyTorch运行协议的有限兼容边界，不能扩展成任意全局配置收集器。
未来替换旧Gym运行时并完成数值回归后再评估移除，不在本次更改开关值。
