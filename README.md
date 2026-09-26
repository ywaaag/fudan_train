# Fudan轮腿机器人训练

基于Isaac Gym Preview 4训练自有六DOF树模型策略，评估checkpoint并导出ONNX。
完整闭链MuJoCo验证由独立的`/home/kellen/wheel_leg_sim2sim`仓库负责。

## 从哪里开始

- 修改代码：先读[ARCHITECTURE](ARCHITECTURE.md)，再读目标模块指南；不依赖历史聊天。
- 协作约束：[AGENTS](AGENTS.md)。架构等价重构进行中，不自动恢复长训练。
- 重构进度：[改动与验证](docs/architecture/refactor_progress.md)、[完整验收台账](docs/architecture/completion_audit.md)。
- 模型证据：[动态启停记录](docs/dynamic_start_stop.md)、[模型分支](docs/model_branches.md)。
  文档是对应日期的实验记录，模型编号或reward上涨不等于当前验收通过。
- 运行命令：[COMMANDS](COMMANDS.md)；当前CLI路径保留，已删除的Python旧入口见[迁移表](docs/modules/compatibility.md)。历史续训示例不是新训练授权。
- 给新 Codex 的最小入口：[CODEX_QUICKSTART](docs/CODEX_QUICKSTART.md)。

## 目录与责任

```text
plane/wheel_legged_gym/
  contracts/       配置结构与序列化
  experiments/     显式输入的实验配方
  domain/          命令、奖励、几何、观测与终止计算
  learning/        网络、PPO、rollout存储和训练循环
  evaluation/      验收门槛、动态响应与候选评分
  ports/           工作流所需进程、artifact、通知接口
  adapters/        Isaac Gym、MuJoCo进程、文件及消息实现
  workflows/       评估、导出、报告等用例
  app/             CLI配置与依赖组装
  envs/            现有环境协调器（配置归contracts，实验归experiments）
  scripts/         原训练/评估/GUI入口
plane/tests/       回归与架构测试
plane/export_onnx/ 原ONNX导出/核验入口
assets/            固定机器人资产
plane/logs/        原checkpoint与训练日志，禁止整体清理
plane/outputs/     验收结果与重构证据，保留旧结果
tools/            架构审计与历史监督器薄CLI，实现在app
docs/             模块指南、实验记录、架构图及历史归档
```

目录职责和外部边界以验收台账为准。依赖箭头和允许方向见ARCHITECTURE，
静态边及源码行号见[dependencies.json](docs/architecture/dependencies.json)。
跨仓库进程接口见[process_interfaces.json](docs/architecture/process_interfaces.json)。

## 固定策略接口

观测25D；history五帧125D，oldest→newest；动作6D，left-first。
腿为position target，轮为velocity target，由混合PD产生torque。
训练physics dt=0.005s、decimation=2。完整scale、gains与limits见AGENTS；本轮不调整。

## 只读检查与测试

```bash
cd /home/kellen/fudan_train
python3 tools/check_architecture.py
PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
/home/kellen/anaconda3/envs/fudan_leg/bin/python -m pytest plane/tests -q
```

训练/Isaac评估必须使用`fudan_leg`；MuJoCo公开验证入口使用`robot`。
这些检查命令不启动长训练。完整仿真结果、checkpoint和ONNX等价证据另见重构记录。

## 历史记录

此前根README中的模型快照、旧启动命令和说明完整保存在
[README原文归档](docs/history/README_before_architecture_20260923.md)。
[原AGENTS交接快照](docs/history/AGENTS_before_architecture_20260923.md)保留历史上下文。
历史文件里的“当前/最新”只代表当时记录；以当前源码、配置和具体run manifest为准。
