# 恢复转弯主线（2026-09-20）

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
