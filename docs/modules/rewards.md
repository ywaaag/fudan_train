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

2026-09-25 独立行为实验 `TURN_ENVELOPE`：仅 B 在
`domain/rewards/turn_lean.py` 为 orientation 提供命令驱动的有界 roll 目标；
A 和其他配方继续走冻结旧公式。权重仍为 -100，pitch 项未删。
源码入口、spec、验证和树模型/闭链边界见 `docs/modules/turn_envelope.md`。

`TURN_LEAN_LONG` 是单独授权的联合能力探索：仍用上述 signed-roll 目标公式，
由显式spec决定上限；高度奖励继续比较 `base_height` 与公开 `commands[:,2]`。
0.40m保留任务不会被转弯高度覆盖。不要把 reward 增长写成实际COM下降或
转弯通过；测量与验收入口见 `docs/modules/turn_lean_long.md`。

`cornering_height_skill_v1` 不改变 `_reward_base_height` 公式或其他奖励系数，
只将 `base_height` 从 `compute_reward` 的单项裁剪中排除。原因是目标 0.36m、
实际 0.40m 时 `8*exp(-.04^2/.001)=1.615`，已超过单项裁剪上限1；
原实现此处对高度误差的梯度为零。新配置的 0.40m 保留环境仍用原系数1，
reference 仅作用奇数 ID 保留 cohort。实际有效量级须读本轮 TensorBoard
`Episode/rew_base_height` 与 probe，不得从公式直接推断已学会降高。
