# 保留对称姿态恢复运动（2026-09-20）

## 重复重试纠错与seed37对照

job motion_goal_20260920_164241三轮回退到同一8200，命令/reward/训练seed23均相同，
因此三轮候选8600结果完全一致：69/117、运动分数2.065696、几何最大5.73cm。
这不是三次独立探索，也不是持续累计训练；旧重试逻辑浪费了两轮算力，应明确纠正。

runner新增training-seed参数，单job记录source SHA/spec/seed签名；下一轮若完全相同，
以paused_duplicate_experiment_prevented停止，不再重跑。跨job仍由操作者核对历史，
不宣称具备全仓库自动去重。未通过原因日志改为同时覆盖安全和姿态保留，避免误称均不安全。

下一轮从同一8200，仅训练seed23→37；保持采样、reward、optimizer配方和候选几何
容差相同。评估seed仍19/37/53，用于同协议比较，不是独立留出集。
这是检验不同训练轨迹的受控对照，不保证改善。如果有合格改进候选，继续其checkpoint；
如果回退后来源/spec/seed不变即暂停诊断，不靠相同重跑维持“连续训练”。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_motion_goal.py \
 --recover-motion \
 --recover-from=plane/logs/wheel_legged/Sep20_16-14-37_motion_goal_20260920_155650_r03_turn1/model_8200.pt \
 --high-speed-geometry-floor=.035 --training-seed=37 --max-rounds=3 --max-stagnant=3
```

## 候选补验及从8200继续

原恢复job三轮均从7900、同训练seed23启动，未沿某条改善链累计1500iteration。
旧日志no safe candidate混合了稳定性不合格与几何保留不合格，不能统称“不安全”。
补验r03的8000/8200，完整seed19/37/53、39命令，证据在原job新增
`recovery_candidate_review.json`及`r03_m{8000,8200}_seed{37,53}.json`。

- 8000安全门槛通过、运动93/117，但+4m/s轮心误差6.81cm且速度误差较大，不选。
- 8200安全门槛通过、运动81/117；停车与|vx|≤1.5几何均≤3cm，最大误差8.93cm
  （较7900的11.59cm降低）。只有+4m/s相对旧保留规则超标：3.43cm超过3cm。
- 8200运动误差分数1.3195，7900为1.3110；运动通过数也从84降为81，不能称为速度恢复。
  本次选择8200是几何改善且低速姿态保留的探索性续训，不是证明优于7900的全指标替代。

增加显式`--recover-from`，只接受已使用geometry_symmetry的profile。候选保留可显式设置
`--high-speed-geometry-floor=.035`，仅|vx|≥2允许3.5cm底线，低速仍3cm；已有较大误差
仍只允许参考+5mm。最终全命令3cm验收门槛不变，不把此容差变化说成模型通过。
从8200完整网络/Adam接续，奖励、命令bank、noise、LR配方不改，固定turn1。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_motion_goal.py \
 --recover-motion \
 --recover-from=plane/logs/wheel_legged/Sep20_16-14-37_motion_goal_20260920_155650_r03_turn1/model_8200.pt \
 --high-speed-geometry-floor=.035 --max-rounds=3 --max-stagnant=3
```

修改run_motion_goal.py、test_motion_geometry.py和本文。7项测试通过，确认3.5cm候选
容差不能扩散到停车/低速，缺失命令仍拒绝。每轮仍smoke、500iteration和三seed验收。
本次job：`plane/outputs/motion_goal_20260920_164241`。已完成基线复测和smoke，
正式r01_train存活，全部网络和两套Adam精确继承，initial_iteration=8200。

用户已确认model7900可视化姿态满意，要求从新姿态恢复运动，不退回旧不对称策略。
来源：`Sep20_15-24-28_motion_goal_20260920_152146_r01_turn1/model_7900.pt`。
此前三seed84/117运动项通过，无failure；停车膝点/轮心镜像误差1.46/2.29cm，
±1m/s几何约.7～2cm，−4m/s轮心仍11.6cm。尚无模型同时通过完整运动和几何门槛。

本阶段只续训：保留turn1命令bank、nominal_state=0、实际几何权重−.1、5mm免罚、
.10m Huber尺度、yaw为零时启用，以及原noise/PPO配置。完整继承网络/std及两套Adam。
不增加失败命令采样focus，不同时调奖励/学习率，也不升turn2。

每轮smoke后500iteration，五个checkpoint先seed19筛选，最佳两个seed19/37/53验收。
稳定性仍是硬门槛。按命令保存model7900三seed最差平均几何距离作为固定参考：
已经≤3cm的项目必须继续≤3cm；其他项目最多允许参考+5mm。
这是恢复过程候选保留条件，不是最终验收放宽。最终仍要求所有停车/直行的膝点和
轮心镜像距离≤3cm，并同时通过全部运动门槛。

姿态保留的安全候选按运动误差分数选取，避免几何最大值主导排序、掩盖运动恢复。
最多6轮、连续3轮无改善暂停复盘；原6900/7400只保留作对照，不作为回退来源。
`accepted`初始为null，因为7900尚未完整通过；只有运动与几何同时通过才设置通过点。
完成状态仍待动态切换与sim2sim，不能直接称实机成功。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_motion_goal.py \
 --recover-motion --max-rounds=6 --max-stagnant=3
```

修改tools/run_motion_goal.py，扩充plane/tests/test_motion_geometry.py，新增本文。
6项相关测试通过，覆盖对称奖励语义、原命令保留、低速姿态门槛、已有缺陷不显著反弹、
缺失命令拒绝。git diff --check通过。运行流程先三seed基线，再smoke、正式训练和验收。
本次job：`plane/outputs/motion_goal_20260920_155650`。基线和smoke已完成，正式r01_train
已启动；manifest检查reward_scales/optimizer/command_bank/geometry_symmetry与7900来源
完全相同，全部网络与两套Adam精确继承，initial_iteration=7900。
