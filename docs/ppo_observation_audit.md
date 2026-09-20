# PPO 更新与观测噪声诊断（2026-09-19）

源为探索分支 `Sep19_12-13-14_anchored_wheel_explore_20260919_121307/model_500.pt`。
此前zero40采样续训退化且已停止；本轮先诊断，不继续使用退化末尾。

## 一次真实PPO更新

`tools/audit_ppo_update.py`：512env、seed23，保持训练noise/domain randomization，
先1000步无更新warmup，再48步rollout和一次PPO/encoder更新；不保存新policy。
输出 `plane/outputs/ppo_audit_source500_20260919.json`。

actor梯度norm约2–11，critic约.004–.04，std约.65–1.2。存在global grad clipping，
但没有证据显示critic梯度压制actor。zero组平均normalized advantage约-.428；
这是相对bootstrapped critic目标，不是独立长期return真值，不能据此宣布critic有bug。
各命令actor动作发生更新，没有发现actor没有参与学习。

## 固定命令噪声诊断

`tools/audit_observation_noise.py`：同checkpoint、seed19、16env/命令、25秒/5秒warmup、
level1参数随机化。两组均采样动作，只有观测噪声开关不同。噪声随机数会改变reset/RNG
轨迹，因此同seed不保证初始物理状态逐位相同；本对照为定位线索，不是多seed因果定论。

| 命令 | 采样动作、观测无噪声 | 采样动作、训练观测噪声 |
|---|---:|---:|
| 0 | -.037 | -.002 |
| -.5 | -.527 | -.528 |
| +.5 | +.421 | +.521 |
| -1 | -.955 | -.993 |
| +1 | +1.026 | +1.064 |

均为真实vx均值，不是MAE。结果文件：
`plane/outputs/wheel500_sampled_seed19.json`、`wheel500_sampled_noisy_seed19.json`。
与deterministic正式验收不同，两个文件都标为diagnostic，不能拿来宣称阶段过关。

## 下一受控实验

`EXPLORE_CLEAN_OBS`从同一model500完整续训：仅关闭训练observation noise，保留
参数随机化level1、动作std/Adam、reward、encoder anchor、原20/20/30/30采样。
不采用已失败的zero40采样，不添加真实速度或命令补偿。
64env×1iteration smoke和6项相关测试通过；源全部模型/两套Adam完全一致。

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py --profile EXPLORE_CLEAN_OBS
```

计划追加500到1000，验收600/1000三seed的0/±0.5/±1。关闭观测噪声是有记录的训练
消融，不代表真实传感器无噪声，也不代表解决未来实机鲁棒性；若成功，之后还需独立
噪声鲁棒性测试。本轮仍未完成最终±4m/s、±4rad/s及自然转弯目标。

新增：两个audit工具、`envs/wheel_legged/clean_observation.py`、
`plane/tests/test_clean_observation.py`和本文。
修改：`envs/wheel_legged/policy_experiments.py`、`scripts/train.py`（均在
plane/wheel_legged_gym），及`tools/run_h3_low_speed.py`。已有模型/代码改动保留。

## 已完成：关闭观测噪声没有解决漂移

run `Sep19_12-49-19_explore_clean_obs_20260919_124912` 在持续退化后提前停止，
没有完成计划500次。保存600/700，各三seed共30条验收：分别6/15、3/15通过。
结果为 `docs/data/clean_observation_20260919.json`，没有升级模型。
这说明噪声改变策略行为的诊断线索成立，但直接关闭噪声不是本起点的有效修复。

额外Adam方向审计在 `plane/outputs/ppo_adam_direction_source500_20260919.json`：
20步中18步actor梯度与实际参数更新点积为负，2步为正。没有证据支持“Adam完全不学习”
或“critic梯度压住actor”。这是单rollout、一阶局部loss诊断，不证明长期回报正确。
