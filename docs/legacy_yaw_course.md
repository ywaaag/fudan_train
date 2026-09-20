# 低速转向和命令切换（2026-09-20）

从已验收平移策略 `Sep19_23-08-55_legacy_speed2_stop_20260919_230848/model_1400.pt`
继续最终goal。保留旧模型，不因训练reward上升替换已验收策略。

## 基线实测

原始数据：`plane/outputs/yaw_baseline_20260920/`。
seed19、每命令16env、level1随机化、deterministic、25秒/5秒warmup。
model1400的原地yaw命令−.5/+.5实际约−.033/−.014rad/s，MAE约.467/.514，
四种低速组合转弯的yaw MAE约.391～.504。没有跌倒，但不具备合格转向能力。

增加单次命令切换评估：5秒时切换，不reset，保持历史四帧，仅更新当前帧命令通道；
10秒起统计稳态指标，保留全过程10Hz逐环境响应trace与全程failure/timeout。
seed19下四种平移切换（0→+.5、+.5→0、+.5→−.5、−.5→+.5）稳态通过。
在vx误差带（停车.05，运动.10m/s）内持续至结束，各16env最慢约.61/1.71/1.11/.71秒；
这是0.1秒分辨率观测，未设加速度/jerk门槛，不能据此宣布运动顺滑。
yaw正负切换未能跟踪。摘要：`model1400_transition_summary.json`。

## LEGACY_YAW 第一轮

只改命令采样：20个等权slot，8个zero、8个平移端点（±.5/±1/±1.5/±2）、
4个纯yaw（±.5各2）。即40%zero、40%平移、20%yaw。episode内命令固定，
还未加入组合转弯或episode内切换训练。height=.4，禁止自动升速。

源SHA校验，reward与optimizer配置不变。80env×1iteration smoke通过，全部网络tensor与
两套Adam精确继承，iteration1400，保存1401。采样覆盖/越界拒绝/验收列表/yaw门槛及
旧采样回归共7测试通过，切换统计2测试通过。机器人asset及policy contract不变。

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py --profile LEGACY_YAW
```

该命令启动新实验。已启动的job为`plane/outputs/legacy_yaw_20260920_000851`，
run为`Sep20_00-08-57_legacy_yaw_20260920_000851`。4096env、seed23、1400→1900，
追加500iteration；1500/1900各三seed验收11命令（9平移+2纯yaw），每点共33项。
必须保留全部平移门槛，纯yaw的vx MAE≤.05、yaw MAE≤.10，其他门槛同平移。
训练结束状态及结果以job/status.json、acceptance.json为准。

本轮已结束：1500为25/33；1900仍未全过，seed19下正yaw伴随约+.091m/s平移。
1900恢复了平移并显著改善yaw，作为同阶段继续优化来源，不替换已验收1400。
用户已授权无人值守继续，见[循环规则与实时job](overnight_motion_goal.md)。

新增：`plane/wheel_legged_gym/envs/wheel_legged/legacy_yaw.py`、
`plane/tests/test_legacy_yaw.py`、`plane/tests/test_transition_summary.py`、
`tools/summarize_transitions.py`。
修改：`plane/wheel_legged_gym/envs/base/command_sampling.py`、`legged_robot.py`，
`plane/wheel_legged_gym/envs/wheel_legged/policy_experiments.py`，
`plane/wheel_legged_gym/scripts/train.py`、`evaluate_policy_comparison.py`，
`tools/run_h3_low_speed.py`、`tools/compare_policy_versions.py`。

旧验收JSON不含yaw_mae时，零yaw仍使用abs_yaw；非零yaw必须有真实yaw_mae，禁止把
原地不转误判为通过。旧平移脚本参数保持兼容。切换的稳态gate与响应trace分开解释。
