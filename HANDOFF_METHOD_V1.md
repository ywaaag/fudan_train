# 轮腿训练交接说明：method_v1

更新时间：2026-09-15  
仓库：`/home/kellen/fudan_train`  
分支：`main`

## 1. 当前结论

项目已经从旧的 H3/H7 经验调参线切换到 `method_v1` 分阶段训练线。当前已经完成：

```text
stand              通过
translate +/-0.5   通过
translate +/-1.0   通过
translate +/-2.0   probe 通过，长训练完成
```

当前下一目标是 `translate +/-3.0 m/s`，对应 `--phase=translate --command_level=3`。

最新 checkpoint：

```text
plane/logs/wheel_legged/Sep07_16-10-41_method_v1_translate_20_long_v1/model_5000.pt
```

## 2. 为什么中途重构

旧训练线可以学会 stand 和低速运动，但进入 3.0--4.0 m/s 后反复进入同一个局部最优：机器人通过轮子空转、牺牲 yaw 和姿态来换取线速度 reward。

主要问题是 reward 和 curriculum 的结构，而不是某个 scale 不够大：

1. reward scale 先乘 policy `dt`，随后单项又被裁剪到 `+/-dt`，大量 scale 调整实际被裁剪吞掉。
2. tracking 主要依赖单一 exponential，高速误差较大时梯度衰减，无法从“完全跟不上”状态得到有效方向。
3. slip 使用左右轮平均速度与 base `vx` 比较，纯 yaw 时合法的左右轮差速会被误判为打滑。
4. 旧 contact lists 为空，base/腿部触地没有可靠的 body-level 约束。
5. curriculum 使用聚合 episode 指标，command 切换、早死和低速样本会稀释高速失败。
6. 旧 H3/H7 把 encoder extra learning rate 设为 `0`，history encoder 实际没有更新。

因此继续在 H3/H7 上机械放大 slip/yaw scale，只会在“速度诱惑”和“惩罚恐惧”之间振荡，不能解决收敛问题。

## 3. 重构边界

重构没有改变部署接口和物理资产：

```text
observation:       25D
history:            5 frames / 125D，oldest -> newest
action:             6D left-first
command:            [vx, yaw_rate, height]
physics dt:         0.005 s
policy decimation:  2
asset:              assets/wheel_leg_train.urdf
```

没有引入 Isaac Lab API、HIM 网络、LQR、static trim、command compensation 或新资产。两个参考仓库只作为 reward/curriculum 设计参考。

## 4. method_v1 已实现内容

### Reward pipeline

新增纯 Torch reward math：`plane/wheel_legged_gym/envs/base/reward_terms.py`。

- coarse/fine tracking + Huber gap cost；
- linear error cap `1.0 m/s`，yaw error cap `0.8 rad/s`；
- task reward 使用 upright/height smooth gate，安全惩罚不被 gate 屏蔽；
- reward term 只做一次 dt scaling，method_v1 不使用旧的逐项 dt clip；
- failure termination 是一次性 cost，timeout 不处罚；
- slip 使用逐轮 rolling residual：`-wheel_omega * radius - (vx - wz*y)`；
- 增加 airborne wheel spin、wheel contact history、lateral velocity、torque/power/action regularization；
- 记录 pre-clip torque saturation、wheel contact fraction、slip RMS 等诊断指标。

### 课程和采样

当前 profile 为手动接力式 `method_v1`：

```text
stand
translate: 0.5 / 1.0 / 2.0 / 3.0 / 4.0 m/s
yaw:       0.5 / 1.0 / 2.0 / 3.0 / 4.0 rad/s
combined:  后续阶段，暂不作为当前主线
```

每个阶段通过 `--phase` 和 `--command_level` 显式选择。command 在 episode 内保持，避免中途 resample 改变 credit assignment。采样器使用 per-env segment counter，避免单个 env 连续 reset 后永远采到 zero。

当前限制：

- method_v1 目前是手动 level 接力，尚未实现完整的 per-sign/per-bin 自动晋级；
- `sample_*_fraction` 日志字段仍沿用旧 mode 汇总方式，不应单独作为 method_v1 采样比例依据；
- command-grid 仍需要扩展并单独验证 +/-3/+/-4 端点；
- combined 阶段必须先确认独立 translation/yaw 能力，不能直接把 `4 m/s + 4 rad/s` 当作首个目标。

### Encoder 和 checkpoint

现有 encoder 保持 `125 -> 128 -> 64 -> 3`，method_v1 使用 `extra_learning_rate=1e-5`，并记录 encoder loss、gradient norm 和 parameter delta。

从旧 H3/H7 迁移必须使用 `--resume_mode=policy`：只加载 actor/encoder，重置 critic、PPO optimizer、encoder optimizer 和旧课程状态。旧 reward 语义不能和 normalized_v1 完整续训混用。

## 5. 已完成训练与结果

### Stand

run：`Sep07_14-34-06_method_v1_stand_from_H3_15800_v1`，checkpoint：`model_3000.pt`

```text
episode length:           2002
wheel contact fraction:   0.9992
zero abs vx:              0.0208 m/s
zero abs yaw:             0.0110 rad/s
wheel slip RMS:           0.0482 m/s
torque saturation:        0
forbidden contact:        0
```

### Translate +/-0.5 m/s

run：`Sep07_15-36-07_method_v1_translate_05_probe_v2`，checkpoint：`model_300.pt`

```text
forward error:            0.0202 m/s
reverse error:             0.0254 m/s
wheel slip RMS:            0.0689 m/s
wheel contact fraction:    0.9992
episode length:            2002
```

### Translate +/-1.0 m/s

run：`Sep07_15-51-36_method_v1_translate_10_probe_v1`，checkpoint：`model_500.pt`

```text
forward error:            0.0139 m/s
reverse error:             0.0303 m/s
wheel slip RMS:            0.0754 m/s
wheel contact fraction:    0.9992
episode length:            2002
```

### Translate +/-2.0 m/s

probe：`Sep07_16-02-31_method_v1_translate_20_probe_v1/model_500.pt`  
long run：`Sep07_16-10-41_method_v1_translate_20_long_v1/model_5000.pt`

```text
forward error:            0.0436 m/s
reverse error:             0.0460 m/s
wheel slip RMS:            0.0623 m/s
wheel contact fraction:    0.9987
episode length:            2002
forbidden contact:         0
torque saturation:         0
action clip:               0
```

这些结果表明当前 policy 在训练分布内没有出现旧 H7 的明显退化，但还不能等价于 +/-2.0 command-grid 已完全通过。

## 6. 当前目标和下一步

### 当前目标

验证并训练 `translate +/-3.0 m/s`：

```text
phase:         translate
command_level: 3
target range:  [-3.0, +3.0] m/s
```

### 推荐 probe 命令

```bash
cd /home/kellen/fudan_train/plane
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/kellen/fudan_train/plane
export CUDA_VISIBLE_DEVICES=0

python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=4096 \
  --resume --resume_mode=policy \
  --load_run=Sep07_16-10-41_method_v1_translate_20_long_v1 \
  --checkpoint=5000 \
  --policy_experiment=method_v1 \
  --phase=translate --command_level=3 \
  --max_iterations=500 --seed=11 \
  --run_name=method_v1_translate_30_probe_v1
```

probe 通过标准：

```text
episode length >= 1900
forward/reverse MAE <= 0.35 m/s
wheel slip RMS <= 0.20 m/s
wheel contact fraction >= 0.90
forbidden contact = 0
torque saturation 不持续贴顶
```

probe 通过后再延长到 `3000--5000` iteration。长训不能只看 mean reward，必须在长训后运行 command-grid，单独验收正向、反向、zero 和接触/滑移。

## 7. 交接注意事项

- 不要把 H3/H7 的旧 optimizer/critic 完整加载到 normalized_v1；
- 不要直接跳到 +/-4 m/s 或同时要求 `4 m/s + 4 rad/s`；
- 不要用 mean reward 或单一 GUI 片段宣布高速成功；
- 不要删除 `plane/logs/wheel_legged/` 和历史 checkpoint；
- 不要修改 `/home/kellen/fudan_rl_wheel_leg/plane`；
- 不要改变 25/125/6 policy contract、URDF、PD、dt 或 decimation；
- 不要把诊断用的 LQR、trim、command gate 混入正式 policy。

## 8. Git 和验证状态

当前主线提交：

```text
f4db39f fix: reset method v1 torque metric buffer
b3d6b90 feat: add method_v1 reward and staged training profile
913ce60 chore: snapshot current wheel-leg training repository
```

最近源码测试：`15 passed, 1 skipped`。

```bash
cd /home/kellen/fudan_train
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/kellen/fudan_train/plane
python -m pytest -q plane/tests plane/wheel_legged_gym/tests
```

历史生成物不进入 Git：`plane/logs/`、`plane/outputs/`、Python cache 和 `egg-info` 均由 `.gitignore` 保留在本地但不版本化。
