# 轮子探索与停车保留（2026-09-19）

最终目标仍为当前URDF的±4m/s、±4rad/s、自然转弯/反向。本轮没有更改验收门槛或资产。

## 已完成轮子探索对照

`ANCHORED_WHEEL_EXPLORE`从低速model100完整续训，只将左轮初始std从0.001255改为
源右轮的0.061858，保持v3 encoder动作锚定、reward/PPO/采样。它只影响训练探索，
部署仍为actor mean，没有速度命令偏移或额外控制器。

run：`Sep19_12-13-14_anchored_wheel_explore_20260919_121307`，100→600。
200/600各三seed；300/400各seed19；500三seed，共11组55条命令验收。
model600的±1三seed通过，但0/±0.5退化。model500更接近完整目标（10/15）。

model500三seed实际vx：

| 命令 | seed19 | seed37 | seed53 |
|---|---:|---:|---:|
| 0 | -0.0576 | -0.0576 | -0.0583 |
| -0.5 | -0.5395 | -0.5521 | -0.5501 |
| +0.5 | +0.3973 | +0.4005 | +0.3981 |
| -1 | -0.9565 | -0.9516 | -0.9579 |
| +1 | +1.0097 | +1.0093 | +1.0085 |

这次±1能力有明显改善，但不是完整±1阶段通过。低速通过基线仍是原model100，
model500只作为更接近目标的实验续训起点。证据含hash与原始指标：
`docs/data/wheel_explore_20260919.json`。

## 停车保留单变量续训

新profile `EXPLORE_STOP_RETENTION`从上述model500完整续训：网络/std/两套Adam逐tensor
校验一致，不再重置任何噪声。仅将原±0.1的20%采样份额转为零命令，zero从20%到40%，
±0.5、±1各保持15%，不改reward或优化器。仍在±1范围内，不是升阶。

64env×1iteration smoke通过，保存model501；8项相关测试通过。实际采样时长仍应由
`command_diagnostics.jsonl`核对，不能把slot比例当作rollout比例。

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py --profile EXPLORE_STOP_RETENTION
```

此命令新建run追加500iteration到1000，自动对600/1000做三seed的0/±0.5/±1验收。
失败则不升级模型。不得因总目标未完成而直接跳到±2/±4。

本轮新增`envs/wheel_legged/stop_retention.py`和`plane/tests/test_stop_retention.py`；
修改`envs/base/command_sampling.py`、`envs/base/legged_robot.py`、
`envs/wheel_legged/h3_speed1.py`、`envs/wheel_legged/policy_experiments.py`、`scripts/train.py`
（这些路径均在plane/wheel_legged_gym内），以及`tools/run_h3_low_speed.py`。
新增本文、上述结果JSON。保留已有未提交改动与全部实验产物。

## 停车采样实验结果：提前停止

run `Sep19_12-31-59_explore_stop_retention_20260919_123152`，从500开始，计划到1000。
实际到约779时连续退化，主动停止训练进程，没有继续追求固定次数。保留600/700；
两点各三seed验收，共30条；未升级checkpoint。

600三seed零命令约-0.131m/s，+.5约+.349m/s，低速误差比源500更大；只有-1通过。
后续训练window的zero MAE升至约.244m/s，而encoder MAE约.018m/s，不能把漂移
主要归因于估计器看不准速度。新采样实际zero占比已接近40%，也不是zero缺样。

原始状态/评估保存在 `plane/outputs/explore_stop_retention_20260919_123152/`，
精简证据 `docs/data/stop_retention_20260919.json`。终态为`paused_on_regression`，
不是完整500次训练成功。source exploration500仍仅是接近完整门槛的实验候选，
正式已通过低速基线仍是Sep19_09-17-38的model100。

没有matched继续原采样的model500续训对照，因此不能严格声称zero40是退化唯一原因；
可以确认这次采样调整未解决问题，不应继续延长失败末尾。下一步需检查actor/critic
的实际优化信号和梯度，区分奖励目标与更新方向的问题，再选单变量实验。
最终±4m/s、±4rad/s及自然转弯目标仍未完成。
