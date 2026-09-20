# 原项目训练配置在当前 URDF 上的对照（2026-09-19）

> 后续源码复核补充：这轮不是严格的原项目配置复现。run manifest仍包含
> `zero_base_velocity=-1`、`zero_wheel_velocity=-1`（原项目没有）；当前legacy
> action_rate按mean计算，原项目按sum计算；PPO encoder梯度裁剪范围也不同。
> 评估器的legacy分支没有累加preclip torque计数，因此其0值不能证明无饱和；
> slip指标使用平均绝对残差，与method_v1逐轮RMS口径不同。
> 这些限制不否定速度读数，但不能据这次对照归因于“原始配置本身偏向前进”。
> 之后若复用此实验，必须先明确隔离这些残留差异，不能仅凭profile名称称其等价。

目的：验证当前 `method_v1`/H3 系列的 reward、command、PPO 和 termination 改动是否是
速度跟踪不佳的唯一原因。该 profile 使用原项目训练语义，但继续使用当前
`assets/wheel_leg_train.urdf`，因此是 **training-config ablation**，不是原项目物理复现。

## Profile 边界

`LEGACY_URDF` 恢复：

- 原始指数 tracking reward：`exp(-error² / tracking_sigma)`，`tracking_sigma=.25`；
- 原始 reward scales、`only_positive_rewards=False`、legacy reward pipeline；
- uniform command/resampling 语义；
- 原始 domain randomization ranges、无训练 push；
- 原始 PPO：actor/encoder LR `1e-3`、adaptive schedule、entropy `.01`、5 epochs、4 minibatches；
- symmetry loss=0，不使用 method_v1 coarse/fine/gap/gate/slip/contact rewards。

保留当前 URDF 的可行 reset 和控制适配：

- asset=`wheel_leg_train.urdf`；
- base z=`0.40`、全零腿角；
- height command=`0.40`；
- current wheel damping=`1.0`，原项目为 `.2`。

因此它回答的问题是：“当前 tree URDF 在原 reward/PPO 训练语义下怎样”，不能回答
“把原项目 infantry asset 完整复现后怎样”。

## 实验结果

命令：

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python \
 plane/wheel_legged_gym/scripts/train.py --task=wheel_legged --headless \
 --num_envs=4096 --max_iterations=500 --seed=23 \
 --policy_experiment=LEGACY_URDF --run_name=legacy_urdf_20260919
```

run：`plane/logs/wheel_legged/Sep19_11-22-46_legacy_urdf_20260919/model_500.pt`。
评估使用同一 `LEGACY_URDF` profile、level-1 randomization、seeds 19/37/53、
16 env/命令、25 s/5 s warmup，固定命令 `0、±0.5、±1`。

| 命令 | 平均实际 vx | vx MAE | 绝对 yaw | 通过 seed |
|---|---:|---:|---:|---:|
| 0 | +0.1073 | 0.1073 | 0.0174 | 0/3 |
| -0.5 | -0.3685 | 0.1315 | 0.0259 | 0/3 |
| +0.5 | +0.5695 | 0.0695 | 0.0164 | 3/3 |
| -1 | -0.8648 | 0.1352 | 0.0286 | 0/3 |
| +1 | +1.0372 | 0.0372 | 0.0144 | 3/3 |

三种子均无失败/timeout，height MAE约1–2 mm，torque saturation为0，wheel slip很低。
原始 JSON：`plane/outputs/legacy_urdf_20260919_seed19.json`、`_seed37.json`、`_seed53.json`。

## 结论

原项目配置在当前 URDF 上确实改变了能力分布：**前进方向（+0.5/+1）明显更好，
但 zero 和 reverse 更差**。所以当前 method_v1 的魔改部分可能是原因之一，但不是唯一
根因；原始配置本身也没有得到完整的双向/停车能力。

本实验还发现当前 legacy evaluator 最初无条件调用 method_v1 wheel residual，已修正为
legacy wheel rolling residual；该工具修复不改变训练 checkpoint。测试和 smoke 通过。

下一步要继续定位，应从这个结果拆出两个单变量实验：

1. legacy reward/PPO + 当前 command sampler；
2. method_v1 reward + 原始 PPO LR/entropy；

都从当前 URDF 的同一 reset/model 起点开始，保留 `0、±0.5、±1` 验收。当前结果不能
支持直接把原项目配置全面替换到主线，也不能继续把 method_v1 的全部改动一起调大。
