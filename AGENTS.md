# Fudan训练仓库协作约束

## 先读当前架构

先读根ARCHITECTURE.md与目标模块指南，任务缺口见docs/architecture/completion_audit.md。
Codex可读性第一；不依赖历史聊天、模型编号或旧快照判断当前状态。

本轮为严格行为等价架构重构：保留现有dirty changes、资产、logs、outputs、checkpoint；
不改变奖励/课程/物理参数/策略契约/验收阈值，不自动恢复长训练。必要1iteration smoke
用于等价验证，不能代替独立验收。短训500iteration规则属于后续明确授权的训练任务。

修改前现场核对Git状态；新增代码优先通过公开模块入口；函数内import也纳入依赖审计。
完成每阶段后更新架构/依赖图与refactor_progress，说明模块、依赖变化和验证证据。
禁止用mixin、动态属性注入或跨仓库monkey patch隐藏依赖；适配器通过显式输入/回调组装。
CLI边界可解析环境输入，domain/recipe不得隐式读取环境变量或文件。

用户2026-09-24补充：旧用法兼容不是硬要求。迁移实际调用方后可删除冗余包装、别名和
旧接口，优先保留清晰的唯一公开入口；变化须记录新入口。训练/评估行为及模型契约仍保持。

历史2026-09-19交接、H系列阶段与旧命令完整归档到
[AGENTS历史原文](docs/history/AGENTS_before_architecture_20260923.md)。
其中当前模型/进程/Git状态不可用作现场事实。模型依据读具体run manifest和验收JSON。

## 1. 项目身份与目录边界

本文件适用于独立仓库：

```text
/home/kellen/fudan_train
```

这是一个基于 Isaac Gym Preview 4 的轮腿机器人强化学习训练仓库，主要负责：

- 训练 six-DOF tree approximation policy；
- 保存和评估 PPO checkpoint；
- 导出 ONNX；
- 为后续 MuJoCo sim2sim 提供策略。

不要把以下目录混为同一个项目：

```text
/home/kellen/fudan_rl_wheel_leg/plane
```

这是 Fudan 原始参考仓库，禁止修改。

```text
/home/kellen/wheel_leg_mjrl-lqr
```

这是旧的综合工作区和 sim2sim 参考目录，保留其中已有改动，不要用它覆盖本仓库。

```text
/home/kellen/wheel_leg_sim2sim
```

这是独立的完整闭链 MuJoCo 验证仓库。训练仓库不要重新实现或临时修改 MuJoCo
闭链控制器来掩盖训练策略问题。

本仓库当前是新迁移出来的独立 Git 仓库，默认分支为 `main`。仓库尚未假设有
远程 origin，也不要未经用户要求执行 push、清理日志或删除 checkpoint。

## 2. 物理和策略边界

训练使用：

- `assets/wheel_leg_train.urdf`：六个 scalar DOF 的树模型近似；
- `assets/wheel_leg_train_parity.xml`：由 URDF 生成的 MuJoCo parity tree；
- `wheel_legged_gym/`：Isaac Gym 环境和 Fudan 风格 PPO；
- `meshes_mj/`：URDF visual mesh 资源；
- `wheeled_infantry.xml`：完整闭链模型，仅作为资产生成和独立 sim2sim 参考。

固定 policy contract，不要随意改变：

```text
observation: 25D
history: 5 frames = 125D，oldest -> newest
action: 6D，left-first
action order:
  [left_leg_0_position,
   left_leg_1_position,
   left_wheel_velocity,
   right_leg_0_position,
   right_leg_1_position,
   right_wheel_velocity]
physics dt: 0.005 s，Isaac Gym 默认 200 Hz
policy decimation: 2，policy 100 Hz
position action scale: 0.5
wheel velocity action scale: 10.0
Kp: [20, 20, 0, 20, 20, 0]
Kd: [1, 1, 1, 1, 1, 1]
torque limits: [40, 40, 47.294118, 40, 40, 47.294118]
```

`ActorCriticSequence` 的网络结构为：

- encoder：`125 -> 128 -> 64 -> 3`；
- actor：`25 + 3 -> 128 -> 64 -> 32 -> 6`；
- critic：训练时可接 privileged observation 和 latent，不能把 critic 输入误当成部署输入。

动作不是直接 torque policy。腿部 action 是 position target，轮子 action 是
velocity target，最终由混合 PD 生成关节 torque。不要把这个接口改成端到端 torque，
除非用户明确重新冻结一套新的 contract。

## 3. 环境启动

必须使用旧版 Isaac Gym 环境：

```bash
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/kellen/fudan_train/plane
export CUDA_VISIBLE_DEVICES=0
cd /home/kellen/fudan_train/plane
```

当前已知运行时：

- Python 3.8；
- PyTorch 2.4.1 + CUDA 11.8；
- Isaac Gym Preview 4；
- 单卡默认使用 GPU 0（RTX 4080）。

不要在 `issaclab`、`isaac` 或 Isaac Sim 新版本环境中运行这里的
`wheel_legged_gym/scripts/train.py`。如果出现：

```text
libpython3.8.so.1.0: cannot open shared object file
```

通常是没有设置 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:..."`，不是 Isaac Gym 安装损坏。

首次安装 editable package：

```bash
python -m pip install --no-deps -e .
```

不要用 pip 自动升级 Isaac Gym、PyTorch 或 CUDA 相关组件。

## 4. 训练入口和日志规则

训练入口：

```bash
python wheel_legged_gym/scripts/train.py
```

完整命令、TensorBoard、导出和评估示例见：

```text
/home/kellen/fudan_train/COMMANDS.md
```

每次实验必须：

1. 使用新的 `--run_name`；
2. 记录 seed、checkpoint、`policy_experiment` 和实际 reward 配置；
3. 先运行小规模 smoke；
4. 再运行正式训练；
5. 用 Isaac command-grid 验收；
6. 导出 ONNX 并做数值一致性检查；
7. 最后才进入独立 MuJoCo sim2sim。

不要删除 `logs/wheel_legged/`。其中包含：

- `model_*.pt`；
- TensorBoard event；
- 本次运行保存的 `wheel_legged_config.py`、`legged_robot.py`；
- `policy_experiment.json`；
- 复现实验所需的 optimizer 和课程信息。


## 7. 对称性问题的正确处理

曾经的问题是左右腿长期不对称。已确认不能简单地提高 raw action symmetry reward。
原因包括：

- 左右物理关节轴、canonical sign 和 body frame 语义可能不同；
- 平衡过程中需要短时左右差动来纠正 roll、接触和质量中心误差；
- 强制 `left_action == right_action` 会把补偿转移到另一条腿，导致更严重的不对称；
- 训练树模型的 COM 本身存在小横向偏置，物理最优控制不一定是逐时刻完全对称。

当前已验证的 wheel-leg URDF 顺序为：

```text
left_leg_0, left_leg_1, left_wheel,
right_leg_0, right_leg_1, right_wheel
```

不要对所有右腿通道统一乘 `-1`。如果要增加 symmetry regularization，必须同时镜像：

- angular velocity；
- projected gravity；
- yaw command；
- 左右 leg position channels；
- 左右 wheel velocity channels；
- previous action；
- 五帧 history。

并通过 mirrored observation -> mirrored action 的 equivariance 约束实现，而不是直接
惩罚同一时刻左右 action 差值。


## 10. 验收标准

单纯满足以下任一项都不能宣布成功：

- mean reward 上升；
- GUI 看起来没有立即倒下；
- episode length 较长；
- 某一个方向的速度看起来正确。

至少需要同时检查：

- zero command 的真实 `base_vx` 接近 0；
- forward 和 backward 方向正确且误差可接受；
- yaw `+/-` 方向正确；
- 目标高速附近有足够样本；
- survival fraction 足够高；
- `curriculum_window_passed=1`；
- 没有持续 torque saturation；
- 左右 wheel 接触稳定；
- policy checkpoint、ONNX 和 sim2sim 输入输出 contract 一致。


## 11. 修改规则

- 使用 `apply_patch` 修改代码；
- 修改前先查看当前 Git 状态和 run manifest；
- 不覆盖用户已有的 dirty changes；
- 不删除 logs、outputs、legacy checkpoints；
- 不修改 Fudan reference；
- 不把诊断用 gravity compensation、LQR、static trim 或 guard 混入正式 policy；
- 修改 reward 后必须做 smoke 和至少一次指标检查；
- 最终报告必须列出确切修改文件、运行命令和验证结果。

如果用户只要求诊断，禁止顺手修改训练代码。只有用户明确要求“调整训练”或
“实现目标”时，才修改课程、采样、reward 或 PPO。
