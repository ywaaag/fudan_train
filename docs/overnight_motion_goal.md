# 无人值守训练（2026-09-20）

用户已明确授权睡眠期间持续训练、验收并朝goal推进。
本次job：`plane/outputs/motion_goal_20260920_002043`。
实时状态：`status.json`；逐轮决策：`progress.md`；每个训练/验收子进程有独立log。
不要重复启动；先检查status中的supervisor_pid和child_pid是否存活。

## 已完成结果与进步原因复盘

截至2026-09-20 04:41，完成30轮正式短训，共15,000 iteration（不含smoke，
包含回退分支，不能当作单条模型继承链长度）。在turns阶段连续三轮未达到配置的
改善幅度后自动暂停：`paused_no_improvement_needs_diagnosis`。现场确认supervisor已退出。

当前已验收模型：
`plane/logs/wheel_legged/Sep20_03-42-29_motion_goal_20260920_002043_r24_yaw4/model_6900.pt`。
SHA256：`24421860582136379edb6e5d9255e8eeb950cfbb97bacd1916921875ed8e2695`。
同一个模型在31个固定命令×3个验收种子中93/93通过，包含零速、前后±4m/s、
原地±4rad/s和低速组合转弯。全部无failure、双轮接触率100%。

| 命令 | 三种子实际平均速度 |
|---|---:|
| +4m/s直行 | +3.99595m/s |
| −4m/s直行 | −3.98710m/s |
| +4rad/s原地转向 | +3.98482rad/s |
| −4rad/s原地转向 | −4.03270rad/s |
| 零命令 | −.01100m/s |

证据：job内`r24_m6900_seed19.json`、`r24_m6900_seed37.json`、
`r24_m6900_seed53.json`；ONNX为`accepted_yaw4.onnx`，256输入一致性检查通过，
最大绝对误差5.7220459e-6，日志`accepted_yaw4_onnx_check.log`。
可机读摘要及模型继承决策见[data/overnight_progress_20260920.json](data/overnight_progress_20260920.json)。

前几天进展有限而这几个小时明显改善，当前证据支持以下解释：

1. **起点不同。** 夜间从已经具备±2平移并开始学会yaw的策略继续；较早实验则尝试
   将站立策略迁移到运动，已有能力与待学习任务不同，不能只比较训练小时数。
2. **课程清晰且保留旧能力。** 逐步加入yaw、升速、组合命令；失败时调整命令采样，
   保留历史端点。夜间各轮保持实际混合legacy配方的奖励、PPO、noise及物理接口不变。
3. **通过实际验收选择保存点。** 每轮筛选五个checkpoint，候选做三种子评估，退化回退。
   30轮不是连续成功，也不以reward或最新checkpoint作为通过证据。
4. **此前存在实现失误。** LEGACY_SPEED2_STOP v1的zero_retention硬编码遗漏±1.5/±2，
   runner又覆盖了完整验收列表；因此那轮不是有效的±2对照。修复、覆盖测试和重做均已记录。
   这些失误造成无效比较，不能归结为“强化学习本来就慢”。

以上是组合流程的证据，不是已隔离的单因素因果结论。尚不能证明reward、optimizer迁移、
采样或某个修复单独解释全部改善，也不能把当前hybrid legacy配置称为原Fudan严格复现。
独立验收是训练外运行的固定命令评估，但同一组种子反复用于选checkpoint，
仍存在选模偏差；应再做未参与选择的新种子/扰动复验。三种子是**验收随机种子**，
不是三次独立训练seed。固定命令稳态通过不等于连续速度域、实机或sim2sim通过。

## 当前瓶颈和下一步

最后一轮候选`Sep20_04-33-38_motion_goal_20260920_002043_r30_turns/model_8200.pt`
为123/129通过，但未替代已验收6900。−4m/s同时+.5rad/s时实际yaw仅约+.016rad/s，
另一个−4/−.5组合vx MAE约.104～.106m/s，也略超.10门槛。
该候选分数虽略低于保留源，但改善小于设定.01，因此记录为回退；不应说完全没有数值改善。
下一步先诊断高速后退转弯，不盲目延长；完整切换课程尚未进入，顺滑性和sim2sim未完成。
保留6900为已验收候选，8100为turns阶段诊断来源，不混淆两者用途。

## 循环与边界

`tools/run_motion_goal.py` 每轮：源SHA校验 → 80env×1iteration smoke →
4096env、seed23、追加500iteration → 汇总TensorBoard → 五个保存点seed19筛选 →
最优两个候选各seed19/37/53完整验收 → 选择或回退。
只有稳定性硬门槛和所有当前阶段命令通过才升阶，验收失败的候选仅能在同阶段继续优化。
已通过模型单独记录，任何实验不会覆盖它。通过阶段后导出ONNX并检查256输入一致性。

阶段：yaw±.5 → 低速组合转弯 → yaw±1 → 平移±3 → yaw±2 → 平移±4 →
yaw±3 → yaw±4 → 组合转弯扩展 → 5秒命令切换课程。
所有阶段保留前面所有命令端点。最终组合包含±1m/s配±1rad/s、±2配±1、±4配±.5，
并非要求同时4m/s+4rad/s。每次只改命令课程或采样权重，不改奖励、物理、noise或PPO。
未通过时可提高失败命令采样占比，但不能删除保留端点或放宽验收门槛。

同阶段连续3轮无改善自动暂停并保留诊断证据；最多40轮后暂停复盘，避免无限GPU消耗。
模拟器/NaN/子进程错误立即停止记录，不启动重复任务掩盖错误。
最终还需切换稳态门槛及逐环境≤3秒进入并保持误差带检查；这是额外响应筛选，
不是完整jerk/超调/运动顺滑性认证。即使课程全部通过也停在
`simulation_curriculum_passed_pending_sim2sim_and_smoothness_review`，不自动宣告最终goal完成。

初始优化来源：`Sep20_00-08-57_legacy_yaw_20260920_000851/model_1900.pt`。
该点已有yaw响应，但正yaw伴随vx漂移，尚未全过。保留的已验收平移模型仍为
`Sep19_23-08-55_legacy_speed2_stop_20260919_230848/model_1400.pt`。

## 控制与记录

```bash
cat plane/outputs/motion_goal_20260920_002043/status.json
cat plane/outputs/motion_goal_20260920_002043/progress.md
# 停止本次supervisor及它自己的子进程，保留所有结果：
touch plane/outputs/motion_goal_20260920_002043/STOP
```

新启动命令（历史入口；本job已暂停，此命令会新建任务而非恢复）：

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_motion_goal.py \
 --max-rounds=40 --max-stagnant=3
```

本supervisor不是中断恢复器：若崩溃，应先核对进程、保存点、spec和验收JSON再人工续接，
不要直接重跑上面命令。使用现有全局训练锁防止两套runner同时训练。

新增：`plane/wheel_legged_gym/envs/wheel_legged/motion_goal.py`、
`plane/tests/test_motion_goal.py`、`tools/run_motion_goal.py`、本文。
修改：`plane/wheel_legged_gym/envs/wheel_legged/policy_experiments.py`、
`plane/wheel_legged_gym/scripts/train.py`；与低速yaw/切换工具一起完成7项相关测试，
阶段保留旧端点、目标覆盖和不安全候选拒绝已测试。源完整状态检查还在每次smoke中执行。
