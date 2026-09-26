# Codex 快速入口（训练仓库）

本仓库用 Isaac Gym Preview 4 训练六自由度树模型 PPO，保存 checkpoint、评估并导出 ONNX。
完整闭链验证属于 `/home/kellen/wheel_leg_sim2sim`。先读 `AGENTS.md`、本页、
`ARCHITECTURE.md`，再按任务只读一个模块指南、目标源码及对应测试。
`docs/architecture/completion_audit.md` 是当前验收结论；旧 run 编号和历史聊天不是现场事实。
工具任务先读 [`docs/CODEX_TOOL_INDEX.md`](CODEX_TOOL_INDEX.md)，其中有查看训练效果、
阶段监控、转弯汇总、ONNX、轨迹图、完成通知和架构检查的唯一入口。
本仓库还提供 `codex-tool-discovery` skill（`.agents/skills/codex-tool-discovery/SKILL.md`）：
涉及工具选择或 checkpoint 筛选时先应用它，避免重复实现和误启动历史监督器。
授权训练或评估时继续读 [`docs/CODEX_WORKFLOW.md`](CODEX_WORKFLOW.md)，沿固定阶段调用工具。
当前转弯/降高实验需要人工接管时，直接读
[`CORNERING_SELF_SERVICE.md`](CORNERING_SELF_SERVICE.md) 和其中指向的 `status.json`。

## 现场与只读审计

```bash
cd /home/kellen/fudan_train
git branch --show-current
git log -5 --oneline
git status --short
python3 tools/check_architecture.py
```

审计只解析 AST，包含函数内导入，不导入 Isaac Gym；输出 `cycles`、
`layer_violations`、`simulator_import_violations`、`missing_local_imports`。
`--write docs/architecture/dependencies.json` 会写依赖图，只在源码依赖变化后使用。

## 目录职责与修改入口

| 路径 | 职责；修改入口 |
|---|---|
| `plane/wheel_legged_gym/contracts/` | 配置结构和默认值；改配置字段从这里进入。 |
| `plane/wheel_legged_gym/experiments/` | 显式输入的配方；改实验选项读 `selection.py` 和目标 recipe。 |
| `plane/wheel_legged_gym/domain/` | 命令、奖励、几何、控制、观测等纯计算；改公式进入对应子包。 |
| `plane/wheel_legged_gym/learning/` | 网络、PPO、存储与 runner；改学习算法进入对应实现。 |
| `plane/wheel_legged_gym/evaluation/` | 指标与验收 gate；改阈值进入对应 gate，不能顺带改训练。 |
| `plane/wheel_legged_gym/workflows/` | 评估、候选推进、导出和报告顺序；改用例流程进入目标模块。 |
| `plane/wheel_legged_gym/ports/` | 外部进程、artifact、通知契约；改接口先核对调用方。 |
| `plane/wheel_legged_gym/adapters/` | Isaac Gym、MuJoCo CLI、文件、进程及通知实现；改外部系统边界从这里进入。 |
| `plane/wheel_legged_gym/app/` | CLI 输入、状态、锁、监督器和依赖组装；改任务启停从目标 `main(root)` 进入。 |
| `plane/wheel_legged_gym/envs/` | 固定 step/reset 时序的环境协调；改生命周期读 `base/legged_robot.py`。 |
| `plane/wheel_legged_gym/scripts/` | 训练、评估和播放 CLI；只放可执行入口。 |
| `plane/export_onnx/` | ONNX 导出与数值核验 CLI；改导出读这里。 |
| `plane/tests/` | 等价、入口和架构测试；按目标模块选择测试。 |
| `assets/`, `meshes_mj/`, `wheeled_infantry.xml` | 树模型资产、网格和闭链参考；只读，改资产须独立授权与验收。 |
| `plane/logs/wheel_legged/` | run manifest、TensorBoard、`model_*.pt` checkpoint；只读旧 run，禁止清理。 |
| `plane/outputs/` | 验收 JSON、轨迹和重构证据；新任务用新路径，禁止覆盖旧结果。 |
| `docs/` | 架构、模块指南、审计台账和历史记录；架构变化更新对应指南。 |
| `tools/` | 仓库审计及历史 CLI 薄入口；改监督器实现进入 `app/`。 |

## 按任务阅读

| 任务 | 接着读 |
|---|---|
| 奖励/课程 | `docs/modules/rewards.md`、`domain/rewards/` 或 `domain/commands/`、目标 tests |
| 观测/动作/网络 | `docs/modules/policy_io.md`、`domain/observations/`、`domain/control/`、`learning/` |
| step/reset/Gym 张量 | `docs/modules/environment_lifecycle.md`、`docs/modules/simulation_tensors.md`、`envs/base/legged_robot.py` |
| 监督器/恢复 | `docs/architecture/supervisor_inventory.md`、`docs/architecture/process_review.md`、目标 `app/` 与进程测试 |
| 闭链调用 | `docs/architecture/process_interfaces.json`、`adapters/mujoco/api.py`、外部 `VALIDATION_API.md` |
| 导出/模型验收 | `COMMANDS.md`、`plane/export_onnx/`、目标 run manifest 与验收 JSON |
| 固定高度转弯包络 | `docs/modules/turn_envelope.md`、`plane/outputs/turn_envelope_20260925_162526/README.md`；只按新授权启动训练 |
| 降高/内倾长训 | `docs/modules/turn_lean_long.md`、`plane/outputs/turn_lean_long_20260925_175224/README.md`；先读 status 与最新摘要，勿批量打开日志 |

## 测试和仿真命令

旧 Isaac Gym 环境仅用于训练/Isaac 评估：

```bash
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/kellen/fudan_train/plane
export CUDA_VISIBLE_DEVICES=0
cd /home/kellen/fudan_train/plane
```

完整测试（从仓库根目录执行；测试可能用 fake subprocess 或短小仿真 fixture，
不会启动长训练）：

```bash
cd /home/kellen/fudan_train
env \
PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
/home/kellen/anaconda3/envs/fudan_leg/bin/python \
-m pytest plane/tests -q --disable-warnings --maxfail=1
```

仅在需要重新证明运行时行为时启动 **一次** 1 iteration smoke；先确认 run_name 未存在，
每次换全新名字。该命令启动 Isaac 仿真、训练子流程，写入 `plane/logs/wheel_legged/`：

```bash
cd /home/kellen/fudan_train/plane
env \
PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
CUDA_VISIBLE_DEVICES=0 \
/home/kellen/anaconda3/envs/fudan_leg/bin/python \
wheel_legged_gym/scripts/train.py \
--task=wheel_legged --headless --num_envs=64 --max_iterations=1 \
--seed=11 --run_name=reviewcodex_unique_smoke
```

这里的 `reviewcodex_unique_smoke` 是命令形状示例，实际执行前须替换为唯一名称。
正式训练命令在 `COMMANDS.md`，仅供人工明确授权后的训练任务执行。
架构审查绝不运行 `--max_iterations=500`、`5000`、`10000`、任何 unattended 长训练、
自动恢复训练，也不运行历史 `tools/run_*.py`、`tools/continue_*.py`、
`tools/train_*.py`、`tools/validate_*.py` 或无 argparse 脚本的 `--help`：
这些入口可能启动子进程、仿真或写旧 job。Isaac command-grid、ONNX 导出和 MuJoCo
`validate_policy.py` 也会执行计算并写输出，只在具体评估任务中运行。

## 固定契约与故障定位

观测 25D；五帧历史 125D，oldest→newest；动作 6D，顺序为
`left_leg_0_position, left_leg_1_position, left_wheel_velocity, right_leg_0_position,
right_leg_1_position, right_wheel_velocity`。Isaac 物理步长 0.005 s，decimation 2，
policy 100 Hz；腿为 position target、轮为 velocity target，混合 PD 产生 torque。
完整 scale、Kp/Kd 与 torque limits 见 `AGENTS.md` 和当前配置。

失败先读 pytest 失败栈及目标源码；训练失败读新 run 的 stdout/TensorBoard、
`policy_experiment.json` 和 checkpoint manifest；监督器失败读其
`plane/outputs/<job>/status.json`、`progress.md` 和对应 `<operation>.log`，
再查 `STOP`、`child_pid` 和 checkpoint hash。不要清理或覆盖 `logs/`、`outputs/`、
资产、checkpoint、`.deep-copilot/`；不修改 Fudan 原始参考仓库。
策略比较的旧 manifest 不能跨源码迁移恢复；`runner_sha256` 现在校验 app 实现，
hash 不符应新建 job，见 `docs/policy_version_comparison.md`。观测噪声诊断沿用调用者
工作目录解释相对参数；registry 导出会覆盖固定 `docs/data` 文件，架构审查不得执行。

训练 smoke：`plane/outputs/architecture_refactor_20260923/final_smoke.log`；
checkpoint/optimizer 对照见 `docs/architecture/refactor_progress.md`；
ONNX 对照：同目录 `final_onnx_equivalence.json`；sim2sim 正常/保护拒绝对照：
同目录 `closed_success_rejection_equivalence.json`；原 dirty 状态保全：
`preexisting_git_snapshots.json`。这些是重构等价证据，不是新策略运动能力验收。
