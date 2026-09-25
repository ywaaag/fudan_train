# 奖励模块最小阅读指南

职责：从显式状态输入计算单个奖励，不负责仿真步进、终止、采样、日志或模型更新。

入口：`wheel_legged_gym.domain.rewards.api.evaluate(name, inputs)`。
`RewardInputs`枚举37个字段，包含当前张量、历史张量、索引及配置；对象冻结，张量引用借用，
不允许奖励修改输入。少数未启用任务的字段可为None，实际使用时仍要求存在。
`EQUATIONS`为只读映射，支持名称与原环境方法一一对应，未知名称不静默降级。

依赖方向：Isaac环境 → rewards.api → equations/inputs/terms → geometry及PyTorch。
该模块不依赖Isaac、CLI、配方选择器或任务注册器。环境原 `_reward_*` 方法保留显式三行
委托，保证已有奖励查找顺序不变，不引入mixin、动态方法注入或callback链。

修改公式前先读inputs及该公式；若增加输入，更新显式字段和环境组装位置。
不能在公式内读取环境变量或仿真对象。配置含义和公式变更不属于本次等价重构。

验证：`PYTHONPATH=plane python -m pytest plane/tests/test_reward_equivalence.py -q`。
fixture来自重构前54个函数的四种配置分支，共216项，使用seed43的非零状态覆盖正负命令、
reset、轮接触和几何；逐项精确比较并断言输入张量未修改。fixture不应随实现自动刷新。
结构移动保留浮点运算顺序；发现语义bug应另立实验，不能改fixture来掩盖差异。
