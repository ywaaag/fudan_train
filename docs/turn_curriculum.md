# 恢复转弯主线（2026-09-20）

## 当前策略依据与验收重点

运行job：`plane/outputs/motion_goal_20260920_135435`，source为6900，height=.40。
初始turn1三seed102/117通过，失败不是只有yaw不足：
前进1m/s配±.5/±1rad/s时，实际vx约1.176～1.235m/s，普遍超速；
后退−1配+.5时，实际yaw约+.353rad/s（MAE .147）。其他三个后退组合较好。
八个新增组合均无failure，双轮contact100%，action_clip和preclip saturation为0。
这些证据仅针对此低速基线，不证明高速没有执行器限制；也不能由此推断唯一训练根因。

训练策略保持以下约束：

1. 同时检查vx与yaw跟踪，不能“转起来”但车速偏离就算成功；四种方向组合等权加入。
2. 低速先采用±1m/s与±.5/±1rad/s，再逐步加±2、±3、±4。高速先小yaw .25/.5，
   避免直接同时要求大线速度和大转速。平面圆周运动的横向加速度量级为|vx*yaw|，
   这只是课程难度安排依据，不是该机器人稳定极限的测量。
3. 保留31个旧命令，失败端点可增加采样，但不删除停车、直行或纯yaw样本。
4. 每500iteration检查五个点，以三seed固定命令门槛升阶。未通过但分数改善的点
   仅是同阶段训练来源，不替换accepted基线；所有完整阶段通过点单独保留。
5. 稳态转弯通过后，需补未知seed、未训练中间命令、左右转向切换及前后反向验收；
   不能将离散固定命令通过称为全部转弯顺滑或实机成功。

本轮不改reward/PPO，不改轮子控制参数，不做隐藏差速补偿，先完成已启动的受控课程。
用户确认继续转弯；40/35cm双姿态需求暂缓，不混入本轮训练。

用户暂停多高度。高度代码/诊断快照提交5855491，11项测试通过；全部模型保留。
转弯来源固定为`Sep20_03-42-29_motion_goal_20260920_002043_r24_yaw4/model_6900.pt`，
不是高度模型，也不是未全部通过的turns8100/8200。height固定.40m，原reward/PPO/
randomization/控制接口保持，完整网络与Adam继承逐项验证。

上次广范围turns最后123/129，−4m/s配+.5rad/s几乎不转；不重复直接跳入全部高速转弯。
新增命令课程turn1→turn2→turn3→turn4：

- 每阶段完整保留yaw4的31个唯一命令：停车、±4直行/自转及低速转弯。
- turn1新增±1m/s配±.5/±1rad/s四象限。
- turn2保留前面命令，再加±2配±.5/±1。
- turn3再加±3配±.25/±.5；turn4再加±4配±.25/±.5。

每轮500iteration，先80env一iteration smoke，筛选五个checkpoint，最优两点三种子
验收。不放宽门槛，不用隐藏补偿。连续3轮无改善暂停诊断；本次最多12轮。
稳态组合通过后仍需动态转弯、反向切换、未知seed和sim2sim；不自动宣称goal完成。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_motion_goal.py \
 --turn-curriculum --max-rounds=12 --max-stagnant=3
```

入口先对原6900做新阶段三seed基线，然后自动smoke/训练/验收。
状态在新motion_goal job的status.json/progress.md；STOP文件只停本流程及其子进程。
修改motion_goal.py、run_motion_goal.py、test_motion_goal.py；新增本文。
