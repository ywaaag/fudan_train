# 优先停车、直行与原地自转（2026-09-21）

用户明确优先五项：停车、前进4m/s、后退4m/s、顺/逆时针原地4rad/s。
组合转弯暂缓，高度固定40cm，对称约束保留。

来源：`Sep21_13-10-28_motion_goal_20260921_124217_r03_envelope1/model_9000.pt`。
原三seed结果：−4m/s约−4.009，+4m/s约3.8975（MAE约.1025，略超.10门槛）；
原地−4rad/s约−4.0346，但seed19/37平移MAE .05052/.05007超.05；
原地+4rad/s约4.0618，停车vx约.0347。五项合计10/15运动门槛通过；
直行/停车几何均通过3cm，保留全部旧数据，不放宽最终门槛。

本轮`basic_motion`只改变命令bank：50个等权槽，其中五个重点各6槽（合计60%）；
14槽保留±.5/±1/±1.5/±2/±2.5/±3/±3.5直行，6槽保留±.5/±1/±2自转。
没有任何vx与yaw同时非零命令。不把新模型解释为保留组合转弯能力。
同源9000完整actor/encoder/critic/std/两套Adam，reward/PPO/noise/物理不改。

先smoke，再每轮500iteration；五checkpoint筛选，最佳两个三seed验收。
25个唯一命令×3seed=75项（其中重点15项）。停车/纯自转vx MAE≤.05，运动vx MAE≤.10，
yaw MAE≤.10，高度MAE≤.03，接触≥.99、无failure/timeout/非轮触地，饱和≤.01。
直行/停车膝点/轮心镜像均值≤.03m。最终全过才接受；不足则可在保留姿态/安全前提下
从改进点继续，并增加失败命令样本。最多12轮，连续3轮无改善或重复配置即暂停诊断。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_motion_goal.py \
 --recover-motion --basic-motion \
 --recover-from=plane/logs/wheel_legged/Sep21_13-10-28_motion_goal_20260921_124217_r03_envelope1/model_9000.pt \
 --high-speed-geometry-floor=.03 --training-seed=23 --max-rounds=12 --max-stagnant=3
```

这轮首先验证稳态五项及中间命令，不代表高速急停距离、.5秒反向切换或实机通过。
修改motion_goal.py/run_motion_goal.py；新增test_basic_motion.py及本文，6项相关测试通过。
本次job：`plane/outputs/motion_goal_20260921_143444`。三seed基线和smoke完成，
正式r01_train已启动。manifest核实50命令槽，reward/optimizer/geometry配方与来源一致，
网络和两套Adam完整继承，initial_iteration=9000。
