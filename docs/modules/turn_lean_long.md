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
| 姿态奖励 | `domain/rewards/turn_lean.py`、`domain/rewards/equations.py` | 使用由四象限几何确定的有界 signed inward roll（与 vx*yaw 反号）；pitch 与原权重保留 |
| 里程碑 | `app/turn_lean_monitor.py`、`tools/monitor_turn_lean_long.py` | 每 1000 step 摘要+轻量探针，250 step checkpoint，STOP 仅停止本 run |
| 阶段/最终评估 | `app/turn_lean_review.py`、`tools/review_turn_lean_long.py` | 原25、训练网格、独立 holdout、出弯和慢速反向分开写原始 JSON |
| 汇总/热图 | `tools/summarize_turn_envelope.py --long` | 保留旧 gate 和历史严格标签；新 lean-aware 标签见本轮 `protocol.md` |
| 完成通知 | `app/completion.py`、`tools/training_completion_hook.py` | status 终态只触发报告/分析；无显式 HAPI 目标不发跨会话消息 |

## 坐标系与内倾符号协议（必须先读）

这是训练和 GUI/评估解释的固定协议，不能凭画面或历史记忆重新猜符号：

- 四元数按 `(x,y,z,w)`，`quat_rotate` 把 body 轴主动旋到 world，
  `quat_rotate_inverse` 把 world 重力被动投影到 body。正 yaw 绕 body/world `+Z`
  旋转；前进且正 yaw 时，短窗轨迹的内侧是 body `+Y`。倒车时必须结合实际
  world 速度判断轨迹内侧，不能只读 yaw 符号，也不能把 world `Y` 当作固定内侧。
- 评估器的 roll 定义是
  `atan2(-projected_gravity_y, -projected_gravity_z)`；正 roll 使 body `+Z`
  向 body `-Y` 倾斜，`projected_gravity_y < 0`；负 roll 向 body `+Y`
  倾斜，`projected_gravity_y > 0`。
- 向圆心的 inward roll 在这个模型/评估 convention 下与 `vx*yaw` **反号**：
  `roll_ref = -clip_tanh(atan(vx*yaw/g), roll_limit)`。
  实现位于 `domain/rewards/turn_lean.py`，必须与
  `scripts/evaluate_policy_comparison.py` 的 roll 计算保持一致。
- `roll_target_mae` 的旧评估目标直接调用训练用 `roll_reference`，是同源检查，
  不能独立证明内倾。新符号诊断用 world 短窗轨迹速度的曲率法向与 body `+Z`
  水平投影点积；低速、近直线、失去双轮接触、滑移或 reset 窗口不判方向。
  COM 相对两轮支撑中心位移单列，不拿轮载荷或整圈位移判方向。
  新评估 JSON 若含 `geometry_trace`，`--long` 汇总还要求所有环境独立几何判为
  inward；旧无 trace 的结果标为 `legacy_unmeasured`，不改写其历史通过数。
  命令、四元数、body 轴、轨迹、轮心和 COM 的证据及 CLI 见
  `plane/outputs/turn_geometry_sign_20260926/README.md`。
- GUI 的 `A/D` 是按键命令，不等于肉眼认定的左/右圆心；先用四象限 JSON 和轮载荷
  诊断确认方向，再解释画面。旧 checkpoint 使用旧符号训练，不能用新符号文档解释为已修复。

符号修复后的新训练必须从 smoke 开始；最小 smoke/四象限诊断通过前，不得从旧模型
继续长训，也不得把历史模型的 roll 曲线与新协议混成一份验收。

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

## 2026-09-26 从 R10200 重建主动内倾

新实验入口是
`plane/outputs/inward_cornering_from_r10200_20260926_174110/README.md`；
先读同目录 `protocol.md` 与 `spec_stage1.json`，再读
`stage1_job/status.json` 和最新 `probes/iter_*_summary.json`。
`experiment_id=inward_cornering_r10200_v1` 在现有 `TURN_LEAN_LONG` 配方内
限定两阶段总预算 25000、新阶段1固定 0.40m、倾角上限 3°；旧三阶段
spec 不变。阶段1从精确 R10200 来源恢复，R10200 teacher 只约束
50% basic-motion cohort。`--geometry-trace` probe 和完整 review 记录
world 速度曲率、body 上轴、轮心和 COM；probe 不是三 seed 验收。
汇总完整 review 时用 `tools/summarize_turn_envelope.py --long
--inward-protocol inward_stage1_v1 --job <review>`；该协议只对目标倾角
至少 1° 的主动内倾点要求独立方向、有效窗口覆盖、0.5° 以上实际朝内投影
和逐环境 roll MAE。普通低负荷转弯保留旧运动/安全 gate。

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

## 2026-09-27 aggressive 采样审计与 v2

先读 [`cornering_bug_audit_20260927_092941`](../../plane/outputs/cornering_bug_audit_20260927_092941/bug_audit_report.md)。历史 aggressive run 未设置 `sampling_version`，其 v1 实际 20 槽比例为全环境 anchor 15%、regional 25%、mid 5%、high 5%；regional 只轮换固定 bank，不是 manifest 声称的连续区域。旧模型与 JSON 保持 v1 身份，不能改写成 v2 数据。

新 spec 显式设置 `aggressive_plan.sampling_version=2` 后，40 槽分配才精确达到声明的 15/22.5/10/2.5%，其余 50% 为奇数 ID 的 basic-motion reference cohort。regional 在 spec bank 的最小/最大 magnitude 范围内连续采样，符号由 env 槽位覆盖四象限；其它 bank 以 env 周期偏移错开首个目标。`policy_experiment.json` 记录 `aggressive_sampling_version`、`aggressive_slots`、`effective_command_fractions` 和准确的 `ablation_variable`。本轮仅做过 64env×1 iteration smoke，尚未用 v2 训练出可接受候选。

历史有效 reward 的 `wheel_slip`、`wheel_contact_loss` 和 `high_speed_slip` 均为 0；高负荷旧 gate 数量不能单独证明安全包络扩大。候选比较同时看逐环境 slip/contact 的 lean-aware gate、同协议 evaluator SHA、命令表和 seed。剩余正式训练预算须先冻结明确的奖励/采样变更再使用。

## 2026-09-27 bug-fix-only 接线验收

新 aggressive spec 必须显式写 `sampling_version=2`。缺字段或显式 v1 搭配 v2 比例均在启动前拒绝；历史无版本 spec 只由 `validate_spec(..., historical=True)` 阅读，旧 run source snapshot 保留其原采样。v2 manifest 另写 `sampling_version`、cohort slots、effective fractions、regional range、seed 与 randomization。40 槽比例是每周期精确值；有限 env 数不足整周期时按实际 ID 计数。

`tools/generate_turn_readiness.py` 现在是薄 CLI，`app.turn_readiness` 根据 `protocol_id` 选择必需 group：固定高度 stage1/aggressive v2 不要求 height group，variable-height 要求。单 seed probe、缺 geometry 或逐环境接触/滑移失败不能成为最终接受证据。结束报告、通知送达和人工 reviewed 使用 `app.completion.completion_facts` 分开读取。

`wheel_rolling_terms` 用每轮 `base_vx-yaw_rate*wheel_y` 的 contact-conditioned 残差；旧 `high_speed_slip` 只看平均轮速，不作为转弯 penalty 自动启用。本轮没有定新 slip scale、没有正式长训。旁路证据见 `plane/outputs/cornering_bug_fix_20260927_101156/`。
