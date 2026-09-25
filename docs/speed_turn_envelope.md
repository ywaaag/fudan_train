# 速度—转弯包络长训练（2026-09-21）

## 本轮进展与中断恢复

第一轮8600完整三seed运动135/141通过，仅纯yaw−3/−4失败，safe=true，
最大平均镜像距离6.71cm（最终3cm几何仍未过），posture_retained=true。
同阶段来源更新为该8600，不设置accepted，不称完整包络通过。
第二轮已训练完成至9100，但在r02_m8800_seed37验收时supervisor及child消失，
status.json未写终态。退出原因未从日志确认，不能假称仍在训练。

新增--resume-evaluation-job：仅允许原状态evaluating且原PID均退出，检查source SHA和
末尾checkpoint存在；跳过已完成训练及smoke，复用已完成JSON，仅补剩余评估。
恢复保留旧history、几何reference、stage、focus和训练参数。当前实现只支持该边界，
不宣称任意中断恢复。无checkpoint删除或覆盖，8项相关测试及py_compile通过。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_motion_goal.py \
 --resume-evaluation-job=plane/outputs/motion_goal_20260921_124217
```

已现场确认恢复至r02_m8800_seed37，supervisor/child存活。完成后按原24轮上限及门槛接续。

用户明确调整目标：高度固定40cm，保持±4m/s直行、±4rad/s原地自转；无需4m/s与
4rad/s同时执行。理想平面圆周运动r=|v/w|，a=|vw|，4×4对应16m/s²约1.63g。
该计算不是摩擦极限或实机稳定性证明。候选包络仍需仿真及实机分别验证。

上一轮motion_goal_20260920_202425完成seed37的一轮500iteration，五保存点没有同时
满足安全与姿态保留的候选，未进入完整三seed候选验收；随后重复保护停止，非进程崩溃。
当前仍以geometry model8200为恢复来源，不采用旧明显不对称模型。

新课程保留yaw4的停车、全部±4直行/自转和低速转弯，使用显式联合命令bank，不独立
uniform抽取vx/yaw，不产生4+4组合。分阶段增量：

1. ±1m/s配±.5/±1/±1.5/±2rad/s。
2. ±1.5/±2m/s配±1/±1.5rad/s。
3. 仅前进3m/s配±.5/±1rad/s。
4. 仅前进3.5/4m/s配±.6/±.8rad/s。

后退超过2m/s只训练直线，不做高速后退组合；以后可单独扩展。低速转向目标较旧课程
提高，高速包络收窄。这里是离散目标包络，不是已实现连续抓地模型或在线安全控制器。
几何reward−.1、旧nominal关闭、noise/PPO保持，命令bank是本轮主要变化。

长训练由最多24个500iteration块组成（上限12000iteration，非保证用满）。每块先smoke，
五保存点筛查、最佳两点三seed验收；通过运动与几何门槛才升阶。失败命令可增加采样，
不删除保留样本；重复来源/spec/seed自动拒绝，连续三轮无改善暂停，不盲跑。
沿用候选高速3.5cm容差，最终直行几何3cm门槛不变。

动态目标单列：原地3rad/s在.3秒达到、转向响应.3～.5秒、急停距离1～1.5m、S弯/
左右切换、无内轮离地及可控roll。这些尚未验收，固定命令不等于障碍绕行成功。
前进4→后退4在.5秒内完成意味着平均16m/s²纵向加速度，不与低速切换使用同一保证；
必须记录初末速度、制动距离、超调和接触，不能把所有反向切换都承诺.5秒。
本轮不启动跳跃或高度训练。稳态通过后仍需动态专项、未知seed及sim2sim。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_motion_goal.py \
 --recover-motion --speed-envelope \
 --recover-from=plane/logs/wheel_legged/Sep20_16-14-37_motion_goal_20260920_155650_r03_turn1/model_8200.pt \
 --high-speed-geometry-floor=.035 --training-seed=23 --max-rounds=24 --max-stagnant=3
```

修改motion_goal.py、run_motion_goal.py；新增test_speed_envelope.py、本文。
8项相关测试通过，包络保留前级命令、无高速后退弯、无4+4；diff检查通过。
本次job：`plane/outputs/motion_goal_20260921_124217`；基线和smoke完成，已进入
`r01_train`，source8200完整网络/Adam继承校验通过。当前尚无新训练验收结论。
