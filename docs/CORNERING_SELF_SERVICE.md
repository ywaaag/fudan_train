# 转弯训练自助操作页

给人和新 Codex 的最短入口。当前实验是
`plane/outputs/cornering_height_skill_20260926_122332/`；先读该目录的
`README.md`、`protocol.md` 和 `spec_stage1.json`。不要靠聊天记录判断模型好坏。
当前 Stage 1 从 `model_35250.pt` 起步，最多新增 5000 iteration；它是训练初始化，
不是已经通过的高速转弯模型。原始 checkpoint 和 accepted 模型不得覆盖。

## 1. 先看有没有结束

从仓库根目录执行，以下三条只读，不启动训练：

```bash
cd /home/kellen/fudan_train
JOB=plane/outputs/cornering_height_skill_20260926_122332/stage1_job
jq '{status,latest_iteration,new_iteration_limit,error,run,milestones:(.milestones|length)}' "$JOB/status.json"
rg --files "$JOB/probes" | rg '_summary\.json$' | sort -V | tail -3
```

`status=running` 时，不编辑当前 `spec_stage1.json`、训练/命令/奖励/监控 Python
源码，也不要再启动一个 GPU0 训练。`latest_iteration` 是监控已验的 checkpoint，
可能落后于训练器正在写入的文件。终态读 `completion_report.md`、最后几个
`probes/iter_*_summary.json`，再看必要的原始 JSON；收到通知不等于训练通过。

监控中的 23 命令探针依次是原五项、高度技能六项、中负荷八项、高负荷四项。
快捷查看一个摘要：

```bash
jq '{target_iteration,checkpoint,checkpoint_sha256,retention_passed,
     height_skill_passed,mid_turn_passed,high_turn_passed,
     height_rows,mid_rows,high_rows}' "$JOB/probes/iter_37250_summary.json"
```

把示例 `37250` 换成上一步列出的实际文件名。高度技能 `0.38m` 和尚未训练的
`0.36m` 必须分开看；单 seed、每命令四环境的 probe 只用于筛查。

训练标量摘要（只读）：

```bash
env PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/summarize_training.py \
/home/kellen/fudan_train/plane/logs/wheel_legged/Sep26_12-28-25_cornering_height_skill_20260926_stage1 \
--window 50 --through-iteration 37250
```

`--through-iteration` 用实际 checkpoint iteration，避免晚到的 TensorBoard
event 混进较早 checkpoint。奖励和 mean reward 不能替代 Isaac 行为评估。

## 2. 如何判断有没有进步

按这个顺序比较**同一协议、同一组命令、同一 seed** 的 checkpoint：

1. 原五项及原 25 命令不能退化；失败、reset、缺测都不算通过。
2. 零速、正负 0.5m/s 的 0.38m 目标，实际 root 高度应跟随并能回到 0.40m；
   同时看质量加权 COM、双轮接触、关节余量。
3. `(|vx|,|yaw|)=(2.5,.6),(3,.7)` 的四象限逐点比较 vx/yaw、root/COM、
   roll、滑移、两轮接触；不能把四方向平均数当成四方向都通过。
4. `(|vx|,|yaw|)=(3,1),(3.5,.8),(4,1)` 是已有高负荷边界，不因为某点
   roll 变大就算通过。候选优先安全和旧能力，再看新增通过点。

旧 gate 在 `plane/wheel_legged_gym/evaluation/gates.py`；本轮高度与 lean-aware
附加门槛在实验 `protocol.md`。不为了通过而调低门槛。记录每个候选完整路径和
`sha256sum CHECKPOINT.pt`；计划里程碑不是文件名，不能默认最新 `.pt` 最好。

## 3. 想改善某个问题时改哪里

| 观察到的问题 | 优先修改 | 先读 |
|---|---|---|
| 0.38m 仍降不下去 | **先复制新 spec**，增加独立高度样本或调整下一阶段高度目标；诊断奖励量级后才考虑系数 | `spec_stage1.json`、`docs/modules/turn_lean_long.md`、`domain/rewards/equations.py` |
| 能在低速降高，转弯时不能 | 新 spec 的 `mid_pairs`/高度课程；检查接触、腿几何和同命令三高度对照 | `domain/commands/turn_envelope.py`、`app/turn_lean_review.py` |
| 某一转向 yaw 差或滑移大 | 新 spec 的命令 bank/占比，逐方向评估；不要直接增大倾角 | `domain/commands/resampling.py`、`domain/rewards/turn_lean.py` |
| 高度斜坡/回高时序错误 | `domain/commands/height_skill.py`；改后加测试和 1-iteration smoke | `plane/tests/test_turn_lean_long.py` |
| probe 未覆盖真实目标 | `app/turn_lean_monitor.py` 的新实验 probe；当前运行的监控器不会加载后续源码改动 | `tools/monitor_turn_lean_long.py`、`protocol.md` |
| 指标计算有问题 | `scripts/evaluate_policy_comparison.py`；另建协议版本，不改旧 JSON | `docs/modules/evaluation.md` |

只改目标模块。25D/125D/6D、PD、限矩、资产和 MuJoCo 保护不能为修饰结果而改。
`TURN_LEAN_LONG` 旧 spec 没有 `cohort_plan` 时仍是旧课程；新实验的 cohort 分配
在 `experiments/recipes/turn_lean_long.py` 和 `domain/commands/resampling.py`。
height reward 公式在 `domain/rewards/equations.py`，本阶段仅将 `base_height`
从单项裁剪中排除，没有改公式或 orientation 权重。

## 4. 评估、续训和停止

安全查看工具及完整命令先读 `docs/CODEX_TOOL_INDEX.md`。
`tools/review_turn_lean_long.py` 现按每个转弯点的 `|vx*yaw|` 从 spec 选择
0.40/0.38/0.36m；旧固定高度 spec 仍使用原高度。含 `cohort_plan` 的新 spec
还会单独写出 `height_skill.json` 与 `height_entry_exit.json`；必须等三 seed 完整结果
及 `summarize_turn_envelope.py --long` 汇总后，才运行
`python3 tools/screen_turn_lean_candidates.py REVIEW_DIR --out NEW_RESULT.json`。
该筛选器拒绝缺测、协议不一致和保留能力退化，不会默认推荐最后 checkpoint。
旧 review 若已有 `long_evaluation_summary.json` 且缺 metric/gate SHA，使用
`tools/summarize_turn_envelope.py --long --job REVIEW_DIR --summary-out REVIEW_DIR/long_evaluation_summary_v2.json`
生成旁路索引；不覆盖旧结果。
已有逐点匹配评估命令见
`plane/outputs/high_speed_cornering_20260926/evaluate_candidates.sh`；该脚本绑定
旧模型和旧输出，仅读取作模板，不原样重跑。公开评估入口是
`plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py`；它启动 Isaac 仿真，
`--out` 必须是新路径。原25命令模板为
`plane/outputs/high_speed_cornering_20260926/evaluate_retention.sh`，同样不要原样重跑。

若 Stage 1 确有进步，才从选中的**实际** checkpoint 复制 `spec_stage1.json`
为新的 `spec_stage2.json`，更新 `source_checkpoint`、`source_sha256`、
`source_iteration` 和 `stage.cohort_plan.height_bank`；teacher 路径/SHA 不改。
Stage 2 可在 0.38m 稳定后加入 0.36m，两个新 run 合计不得超过 20000
iteration。每个新配置先 64env × 1iteration smoke，再用新 `run_name` 启动
正式训练；`--max_iterations` 是**本次新增量**。训练必须
`--resume --resume_mode=full`，实际命令见现有 run 的
`policy_experiment.json.training_command`，不能猜参数或误用同编号 checkpoint。
长训监控用 `tools/monitor_turn_lean_long.py`，其 `--job/--run/--spec/
--trainer-pid/--new-iterations` 必须与真实训练进程一致。

若需停止**当前** run，在仓库根目录执行 `touch "$JOB/STOP"`；监控器只会
对其记录的 trainer PID 发 SIGINT，并在 `status.json` 记录原因。不要杀其他
历史训练或清理 checkpoint。终态 hook 只发送完成消息，不会自动续训。

改代码后运行：

```bash
cd /home/kellen/fudan_train
python3 tools/check_architecture.py
env PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
/home/kellen/anaconda3/envs/fudan_leg/bin/python \
-m pytest plane/tests -q --disable-warnings --maxfail=1
```

这些是代码检查，不是运动能力验收。源码和完整命令的本次运行身份见
`plane/outputs/cornering_height_skill_20260926_122332/source_identity.md`。
