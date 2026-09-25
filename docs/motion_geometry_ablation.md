# 转弯后加入真实几何对称约束（2026-09-20）

## 上一轮结果及更正

job `motion_goal_20260920_135435`完成5轮。turn1第二轮7400通过117/117速度/接触/
稳定性门槛；不含姿态对称门槛，不能称整体合格。turn2第三至五轮均无安全候选，暂停。
每轮五个checkpoint seed19筛查合计5次failure、5条记录有全程非轮触地；没有记录
接触率<.99、高度MAE>.03或饱和>.01。此前“没有跌倒”的表述错误。
第一轮95/117虽少于基线102/117，但最坏归一化误差分数下降；应说分数改善而非通过率提升。

## 受控实验

来源：`Sep20_14-09-30_motion_goal_20260920_135435_r02_turn1/model_7400.pt`。
固定height=.40，维持turn1全部39命令、原noise/PPO/optimizer，不继续turn2升速。
将不匹配tree语义的旧nominal_state(-1)关闭，替换为stand_bilateral_geometry(-.1)。
该项计算真实膝点/轮心相对root的位移，逆旋转到机身坐标系，对右侧Y镜像后比较。
Huber误差尺度.10m，5mm免罚区；仅|yaw command|<.01启用，包含停车与直行，
转向不强制几何对称。不锁raw action，不改右腿关节符号，不注入部署补偿。
这是对称reward配方替换，含关闭旧项和加入新项，不能单独归因于任一子改动。
旧critic和Adam完整保留作为初始化；奖励语义改变，不称为同目标精确续训。

以`--geometry-symmetry`单独开启；旧profile默认行为保持。legacy环境显式初始化
膝点/轮心索引，不能假定normalized profile专用索引已经存在。

验收：39命令×三seed，原速度/接触/稳定性门槛保留；停车/直行额外要求膝点与轮心
镜像距离的时空均值分别≤3cm。这是本轮明确的工程目标，不代表实机机械公差；
只有运动与几何都过才算通过，几何改善但速度退化不能替换基线。
每轮500iteration，先smoke；五保存点先seed19，最佳两个追加37/53。
本次只1轮，不自动升速。绝对3cm门槛距离当前27/53cm偏差较远，未预期一轮必然修复。

job：`plane/outputs/motion_goal_20260920_152146`。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_motion_goal.py \
 --geometry-symmetry --max-rounds=1
```

修改legged_robot.py、motion_goal.py、run_motion_goal.py；新增test_motion_geometry.py及本文。
5项相关测试通过：镜像姿态零惩罚、停车/直行启用、转弯豁免、共同机身旋转不变性、
源命令及optimizer不变、旧reward替换范围。运行时还需manifest、smoke和event指标核对。
运行确认：smoke已完成，正式r01_train启动，源iteration7400、所有网络tensor和两套
Adam精确检查通过。manifest核实nominal_state=0、stand_bilateral_geometry=−.1。
