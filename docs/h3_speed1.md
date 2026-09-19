# H3 系列 ±1 m/s 课程（2026-09-19）

源：`plane/logs/wheel_legged/Sep19_09-17-38_h3_low_speed_20260919_091730/model_100.pt`。
该模型已通过三种子 0/±0.5 验收，±1 仍未通过。本轮只扩大命令分布，保留低速回归。

Profile `H3_SPEED1` 必须使用 `--resume --resume_mode=full`。载入前检查已验收源的
SHA256、reward scales/parameters、optimizer、randomization和symmetry；载入后逐tensor
检查完整网络/std、两套Adam状态，iteration必须从100开始。旧分支resume检查保留。

reward、随机化level1、无推扰、actor/critic/encoder LR=1e-5 fixed、entropy=.001、
mirror loss=.01、资产/PD/接口均保持。唯一主要变量是command采样：

| 命令 | 比例 |
|---|---:|
| 0 | 20% |
| -0.1 / +0.1 | 各10% |
| -0.5 / +0.5 | 各15% |
| -1 / +1 | 各15% |

method_v1的episode保持命令与per-env segment counter保留。新增可选正值endpoint
列表，仅该profile启用(.5,1)，在连续20个slot内覆盖上述比例，端点按实际range裁剪。
不启用该列表时原采样器行为不变。比例是采样slot比例，不冒充实际rollout时长比例。

## 启动与验收

64 env、1 iteration smoke已完成：
`Sep19_09-37-49_h3_speed1_smoke_20260919/model_101.pt`。
完整状态继承断言通过，13项sampler/profile/迁移/gate测试通过。

正式任务：`plane/outputs/h3_speed1_20260919_093828/`。
seed23、4096 env，追加500 iteration到model600；评估model200和model600。
每点三seed（19/37/53）、每命令16env、25秒/5秒warmup，deterministic，
固定 `0、±0.5、±1`，门槛与上一轮相同。必须15/15通过才算该阶段通过。

```bash
# 仓库根目录；启动新实验，不是恢复命令
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py --profile H3_SPEED1
```

准确命令和运行目录在任务status.json，checkpoint和manifest留在logs。
中断后先核对进程和checkpoint，不重复执行上述命令。

本轮实现文件：

- 新增 `plane/wheel_legged_gym/envs/wheel_legged/h3_speed1.py`
- 新增 `plane/tests/test_h3_speed1.py`
- 修改 `plane/wheel_legged_gym/envs/base/command_sampling.py`
- 修改 `plane/wheel_legged_gym/envs/base/legged_robot.py`（仅传递可选采样参数）
- 修改 `plane/wheel_legged_gym/envs/wheel_legged/policy_experiments.py`
- 修改 `plane/wheel_legged_gym/scripts/train.py`
- 修改 `tools/run_h3_low_speed.py`
- 新增本文档；结果完成后同步README、模型索引及精简验收证据。

## 完成结果：未通过，保留源模型

正式run：`Sep19_09-38-35_h3_speed1_20260919_093828`。
100→600追加500 iteration完成。200/600各三seed验收；300/400/500补测seed19，
发现失败即无需再用其余seed证明全通过。共9组、45个命令/seed条目。
初始root state、observation、环境配置均与上一轮model100逐项一致。
没有任何已保存checkpoint完整通过，任务为 `paused_on_regression`。

三seed平均实际vx（m/s）：

| 命令 | 源model100 | 本轮model200 | 本轮model600 |
|---|---:|---:|---:|
| 0 | -0.0080 | -0.0113 | -0.0603 |
| -0.5 | -0.4087 | -0.4156 | -0.5005 |
| +0.5 | +0.4304 | +0.4177 | +0.3472 |
| -1 | -0.7999 | -0.7256 | -0.8562 |
| +1 | +1.2643 | +0.9424 | +0.8945 |

model200的低速9/9及+1三seed通过，但-1全部失败（MAE约0.2744），阶段总计12/15。
model600的停车、+0.5和-1均失败，+1仅一个seed通过；不能因后退变准而采用它。
300/400/500在seed19均未通过。所有原始结果、哈希和source manifest见
[本轮证据](data/h3_speed1_20260919.json)，任务目录`selection.json`明确保留源model100。

本轮没有导出新的部署模型，没有运行sim2sim，没有启动下一轮训练。
下一步应从已通过低速的源model100保留基线，先核对按命令分组的实际rollout覆盖、
reward贡献及策略更新变化，定位为何前进/后退与停车出现取舍；不要盲目延长600。
目前不能归因为critic、encoder或源std不对称；需要受控诊断证据。

本轮额外文档改动：`README.md`、`docs/model_branches.md`、
`docs/data/h3_speed1_20260919.json`。原有未提交改动和全部模型保留。
