# TURN_LEAN_LONG：Codex 最小入口

本轮能力探索与 2026-09-25 的固定高度 `TURN_ENVELOPE` 是两个实验。
先读本页，再读
`plane/outputs/turn_lean_long_20260925_175224/README.md`、
`spec_stage1.json` 和当前阶段摘要；不要先扫描完整 stdout 或旧日志。
所有新训练总量最多 30000 PPO iteration、最多 3 个 run；不自动续旧启停任务。

| 职责 | 源码入口 | 当前约束 |
|---|---|---|
| 解析 spec / CLI | `app/experiment_inputs.py`、`scripts/train.py` | `FUDAN_TURN_LONG_SPEC` 仅在 app 读文件；训练必须 `--resume_mode=full` |
| 纯配方 | `experiments/recipes/turn_lean_long.py` | 只收显式 spec，50% 以上为 R10200 basic_motion 保留任务 |
| 来源 / teacher | `adapters/artifacts/recipe_source.py`、`scripts/train.py` | 核对 source 路径+SHA、完整模型及双 Adam；teacher 始终为 R10200 SHA |
| 命令与高度 | `domain/commands/turn_envelope.py`、`domain/commands/resampling.py` | 公开命令使转弯高度在 yaw 加入/退出时 0.40↔目标；保留环境为 0.40 |
| 姿态奖励 | `domain/rewards/turn_lean.py`、`domain/rewards/equations.py` | 使用执行命令 vx*yaw 的有界 signed roll；pitch 与原权重保留 |
| 里程碑 | `app/turn_lean_monitor.py`、`tools/monitor_turn_lean_long.py` | 每 1000 step 摘要+轻量探针，250 step checkpoint，STOP 仅停止本 run |
| 阶段/最终评估 | `app/turn_lean_review.py`、`tools/review_turn_lean_long.py` | 原25、训练网格、独立 holdout、出弯和慢速反向分开写原始 JSON |
| 汇总/热图 | `tools/summarize_turn_envelope.py --long` | 保留旧 gate 和历史严格标签；新 lean-aware 标签见本轮 `protocol.md` |
| 完成通知 | `app/completion.py`、`tools/training_completion_hook.py` | status 终态只触发报告/分析；无显式 HAPI 目标不发跨会话消息 |

训练入口仍是 `plane/wheel_legged_gym/scripts/train.py
--policy_experiment=TURN_LEAN_LONG`；实际完整命令记录于新 run 的
`policy_experiment.json`。当前 stage1 使用 `spec_stage1.json`，R10200 作为来源
和固定 teacher，4096 env、seed23、最多新增10000 iteration。监控当前 job 的
`status.json`，每1000步读 `summaries/iter_<absolute>.json` 和
`probes/iter_<absolute>_summary.json`；完整日志只在定位异常时读对应路径。

旧物理/动作契约、asset、PD、限矩和 MuJoCo 保护均未改变。新评估记录实际
root/质量加权 COM 高度、目标/实际 roll、两轮力、关节软限位余量和 reset 分类；
`evaluation.gates.gate` 仍为旧数值门槛。旧 `abs_roll≤0.10rad` 只作为历史
对照，不用来否定经测量安全的较大主动内倾；新门槛训练前写在本轮 `protocol.md`。

本轮未提交的其他 dirty 文件、旧 logs、outputs、checkpoint 和原始参考仓库不清理。
架构审计：`python3 tools/check_architecture.py`；完整测试用
`fudan_leg` 环境运行 `python -m pytest plane/tests -q`。新增依赖图只在代码
稳定后用 `python3 tools/check_architecture.py --write
docs/architecture/dependencies.json` 更新。

## 2026-09-26 独立降高技能阶段

当前授权实验入口仍是 `TURN_LEAN_LONG`，但只有 spec 含 `cohort_plan` 时才切换为
`cornering_height_skill_v1`。最小证据入口：
`plane/outputs/cornering_height_skill_20260926_122332/README.md`、`protocol.md`、
`spec_stage1.json`。不要将旧 `turn_lean_long_v1` 的固定 50/50 转弯采样或
`turn_height_reward_scale` 裁剪行为混入这个新 run。

20 个环境槽位中奇数为 50% 原 basic_motion，偶数 0/2/4/6 为 20% 独立高度技能，
8/10/12/14/16 为 25% 低中负荷转弯，18 为 5% 原高负荷边界。
`domain/commands/height_skill.py` 仅生成公开 0.40↔目标高度命令；
`resampling.resample` 分配 cohort，环境只组装 scheduler 回调。
R10200 的 reference loss 使用现有 stride=2，仅选择奇数 retention ID。

Stage 1 仅训练 0.38m 高度技能；监控仍探测未训练的 0.36m，必须分开解读。
新 spec 唯一奖励变更是把 `base_height` 加入 `unclipped_reward_names`；高度公式、
8 倍转弯/降高缩放、orientation=-100 和其他奖励权重保持原值。原来 0.04m 误差处
8 倍高度奖励达到单项裁剪平台，无法提供继续降高的局部梯度。
监控的 23 命令 probe 分为保留5、高度6、中负荷8、高负荷4，输出实际 checkpoint
iteration/SHA 和每组失败分类；probe 是单 seed 筛查，不是最终三 seed 验收。
候选 `turn_lean_review` 对包含 `height_schedule` 的 spec 按每个目标命令
`abs(vx*yaw)` 选择转弯高度，原25命令始终0.40m；旧无 schedule 的 spec
沿用固定 `turn_height`。含 schedule 的 `entry_exit` 从训练bank分别选择0.38和0.36目标
转弯，以公开命令检查出弯回到0.40。含 `cohort_plan` 的 review 另生成
`height_skill`（0.38/0.36m、零速及正反0.5m/s）与 `height_entry_exit`；
`summarize_turn_envelope.py --long` 为其分别计算逐环境高度门槛和回高阶段指标。
`screen_turn_lean_candidates.py` 只有在这些组与原25/转弯/holdout均三seed完整、
checkpoint及协议同一时才允许比较候选；单seed probe不进入接受名单。
