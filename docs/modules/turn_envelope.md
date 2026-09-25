# 固定高度转弯实验：Codex 修改入口

先读本页，再读本轮 [`README.md`](../../plane/outputs/turn_envelope_20260925_162526/README.md) 和目标源码；
不要读全部历史日志。训练入口是 `plane/wheel_legged_gym/scripts/train.py
--policy_experiment=TURN_ENVELOPE`，运行命令见根 `COMMANDS.md`。本轮 A/B 已
完成，禁止复用 run_name 或覆盖输出。

| 要修改的职责 | 唯一入口 | 必须同时核对 |
|---|---|---|
| spec 读取、来源验证 | `app/experiment_inputs.py`、`adapters/artifacts/recipe_source.py` | R10200 路径、SHA、reward、完整模型和双 Adam 恢复 |
| 25% 转弯 / 75% 保留采样 | `experiments/recipes/turn_envelope.py`、`domain/commands/resampling.py`、`domain/commands/turn_envelope.py` | 固定 0.40m、公开斜坡、四象限 bank、reference cohort |
| A/B 唯一变量 | `domain/rewards/turn_lean.py`、`domain/rewards/equations.py` | orientation 权重 -100 不变；pitch、零/直行/原地旋转保持水平 |
| 训练更新 | `scripts/train.py`、`learning/modules/policy_retention.py` | encoder 冻结、PPO LR 1e-6、teacher=R10200、只约束 75% 保留环境 |
| 数值评估 | `scripts/evaluate_policy_comparison.py`、`evaluation/gates.py` | 原 gate 不变；新严格转弯标签见本轮 `protocol.md` |
| 证据汇总 | `tools/summarize_turn_envelope.py`、`tools/render_turn_trace.py` | 只读原始 JSON，不插值未测点，不把动画当相机画面 |

`FUDAN_TURN_SPEC` 只在 app 边界读取；recipe/domain 不读环境变量或文件。
`env_id % 4 == 0` 为 turn cohort，标签不进入 25D observation。先随机
等待 0.5–1.5s，再用 2s 加速度、2s 加 yaw，随机保持 2–5s，最后 2s 出弯；
全程只由公开命令驱动，没有速度反馈补偿。保留 bank 为 R10200 的 50 槽
`basic_motion`；转弯 bank 为 `|vx|={.5,1,2}`、`|yaw|={.25,.5}` 四象限。

B 使用命令参考 `2° * tanh(atan(vx*yaw/9.81)/2°)`，A 仍用原水平参照。
URDF 零关节两轮中心间距约 0.441m；2° 对应简单几何高度差约 15.4mm，
略高于独立闭链 15mm 腿长差保护。本轮只验证 Isaac 树模型，不修改保护或推断
闭链可迁移。B 的实际 roll 未达到参考，不能把小幅数值变化解释成主动内倾成功。

最小检查：`python3 tools/check_architecture.py`；`fudan_leg` 环境运行
`python -m pytest plane/tests/test_turn_envelope.py -q`。完整回归见根
`docs/CODEX_QUICKSTART.md`。评估 CLI 默认保留旧单斜坡时序；新选项
`--yaw-delay-seconds`、`--yaw-exit-at`、`--yaw-exit-factor -1` 只在显式传入
时启用。每份原始 JSON 都记录执行命令、策略观测命令通道、history 最新帧、
真实运动/轨迹、接触及 reset；动态段汇总与稳态 gate 独立。
