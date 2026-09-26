# 架构目标验收台账（2026-09-25）

本表只验收架构重构的行为等价，不宣称某个 policy 已通过新的运动能力 gate。
`已完成` 指此项的当前架构目标有源码、测试或代表性运行证据；
`明确边界` 指外部运行时或历史格式仍需具体任务验证；`未完成` 指尚缺当前目标所需证据。
本表记录的 2026-09-25 架构阶段没有改奖励、课程、物理、模型、策略 contract
或验收阈值，也没有恢复长训练；其后授权的运动训练是独立行为变更，不属于本表等价结论。
现场依赖审计输出和实际测试结果在 `refactor_progress.md` 最末阶段；
逐入口动态审查见 `process_review.md`。命令均从对应仓库根目录执行。

| 项目 | 当前状态 | 证据文件 | 验证命令 | 剩余边界 | 影响架构 Goal 完成？ |
|---|---|---|---|---|---|
| 1 仓库自描述 | 已完成 | `README.md`、`ARCHITECTURE.md`、`docs/CODEX_QUICKSTART.md`；sim2sim 同名入口 | 两仓库 `git status --short`，按快速入口核对目标文件 | 文档须随入口变化更新 | 否 |
| 2 循环依赖 | 已完成 | `dependencies.json`；sim2sim `dependencies.json` | 两仓库 `tools/check_architecture.py` | AST 不涵盖反射和第三方内部 import | 否 |
| 3 单向分层 | 已完成 | `ARCHITECTURE.md`、`dependencies.json`、`plane/tests/test_architecture.py` | 训练 `python3 tools/check_architecture.py`；sim2sim 用 robot Python 运行同名脚本 | sim2sim legacy CLI 仅审计循环，允许边白名单只覆盖核心模块 | 否，已明示范围 |
| 4 环境模块化 | 已完成 | `docs/modules/environment_lifecycle.md`、`simulation_tensors.md`、`plane/outputs/architecture_refactor_20260923/final_smoke.log` | 训练全套 pytest；必要时 64 env/seed11/1 iteration smoke（命令见快速入口） | Isaac Gym 与 GPU 运行时属第三方 | 否 |
| 5 工作流模块化 | 明确边界 | `docs/architecture/process_review.md`、`supervisor_inventory.md`、`plane/tests/test_compare_policy_entrypoint.py`、`test_job_tool_entrypoints.py` | 训练全套 pytest；临时目录/fake subprocess 验证定向入口 | 比较、registry、噪声诊断缺陷已修；其余历史 job 的真实文件协议和 GUI/CLI 运行路径尚未逐项证实 | 是，完整行为等价结论暂缓 |
| 6 隐式全局状态 | 已完成 | `docs/modules/configuration.md`、`compatibility.md`、`plane/tests/test_config_instance_isolation.py` | 训练全套 pytest | 第三方 Torch/Isaac 开关、GUI 输入与历史 job 格式 | 否 |
| 7 跨仓库公开接口 | 已完成 | `docs/architecture/process_interfaces.json`、sim2sim `VALIDATION_API.md`、`plane/tests/test_cross_repository_boundary.py` | 两仓库全套 pytest；两仓库依赖审计 | 子进程环境、PID 与文件系统故障是外部边界 | 否 |
| 8 单一公开入口 | 明确边界 | `docs/modules/compatibility.md`、`docs/architecture/process_review.md`、sim2sim `policy_measurements.py` | 两仓库全套 pytest；定向 CLI 参数绑定测试 | 旧 Python import 不再承诺；其余实际 CLI/root 路径尚需按用途核对 | 是，不能从 import 成功推断 CLI 可用 |
| 9 训练行为等价 | 已完成（代表性 smoke） | `plane/outputs/architecture_refactor_20260923/final_smoke.log`、`config_isolation_smoke_equivalence.json`、`docs/architecture/refactor_progress.md` | 快速入口的 1 iteration smoke；训练全套 pytest | 不覆盖全部运行分支、GUI 或完整课程；不等于 command-grid 能力验收 | 是，对“全仓库等价”仍缺证据 |
| 10 ONNX 等价 | 已完成 | `plane/outputs/architecture_refactor_20260923/final_onnx_equivalence.json` | 读取该 JSON 的输入/输出 shape、hash、batch1/8/256 `outputs_exact`；导出命令见 `COMMANDS.md` | 只覆盖记录的 checkpoint，不能推广到任意模型 | 否 |
| 11 sim2sim 等价 | 已完成（固定步长路径） | `plane/outputs/architecture_refactor_20260923/closed_success_rejection_equivalence.json`、tree 固定步长对照见 `refactor_progress.md` | robot Python 测试；公开 34000 步命令见 sim2sim `CODEX_QUICKSTART.md`，本轮不重跑 | 正常 2000 步与保护拒绝 2489 步分别解释；拒绝不算能力通过 | 否，能力验收另案 |
| 12 Git 可追溯 | 已完成 | `plane/outputs/architecture_refactor_20260923/preexisting_git_snapshots.json`、`docs/architecture/refactor_progress.md` | 两仓库 `git branch --show-current`、`git log -5 --oneline`、`git status --short` | 当前工作区状态以现场 Git 输出为准；训练 logs/outputs 不随源码提交 | 否 |

## 明确边界

- **第三方运行时**：Isaac Gym Preview 4、PyTorch/CUDA、MuJoCo、操作系统进程/PID、
  GUI/pynput 和 HAPI 通知不由本仓库静态审计证明；失败应读对应新 job 的 status/log。
- **历史 job**：`process_review.md` 逐项说明 lock、STOP、PID、hash 和恢复能力。
  无显式恢复的入口不能自动续训；历史文件格式只在实际需要恢复时核对。
- **模型能力**：zero、forward/backward、yaw±、高速样本、survival、
  `curriculum_window_passed`、torque saturation 与双轮接触必须通过独立 command-grid
  和闭链验收。已有等价证据只证明重构未改变记录路径，不授予新模型通过结论。

## 未完成

上一版“自有模块项全部关闭”过度概括。静态分层、单元测试和代表性 smoke 有证据；
本轮又通过定向测试修复三个真实 CLI 缺陷。仍需：用隔离的历史 job 样本验证读写与
恢复拒绝路径；核对其余高价值 CLI 的 root/参数和输出位置；对 GUI 交互做受控验证。
完成这些定向验收前，不建议宣布“全仓库运行路径严格行为等价”或完成整个架构 Goal。
新 policy 的完整运动能力 gate 还需要独立授权和评估，不能用本表替代。
