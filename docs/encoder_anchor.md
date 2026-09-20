# Encoder 动作锚定实验（2026-09-19）

以已验收低速model100为源，沿用H3_SPEED1采样/reward/PPO配置，仅新增encoder监督
loss上的动作MSE锚定，coef=1。参考encoder在本轮PPO actor更新完成后、encoder更新前
复制，actor权重在encoder optimizer阶段不被step。约束是soft penalty，不是硬KL上限。

## 版本及已完成证据

- v1 smoke的参考encoder取自当前minibatch，loss为0，已判为无效实现，未正式训练。
- v2修复为固定encoder快照，完成500iteration及30条验收。
  run：`Sep19_11-44-09_encoder_anchored_20260919`。
  model200 3/15通过，model600 10/15通过。model600真实vx三seed均值：
  `0:-0.0249、-0.5:-0.4423、+0.5:+0.4009、-1:-0.7974、+1:+0.9662`。
  encoder引起的轮子动作偏移末100次约0.00112/0.00090，probe KL约0.359，低于
  updating对照的约124；但未通过所有速度门槛。
- v2复核发现anchor用了next observation与current history配对。其软约束仍有数值
  效果，但不能声称约束的是实际同一时刻输入。保留为有缺陷的诊断结果，不升级模型。
  数值证据：`docs/data/encoder_anchored_20260919.json`。
- v3修复为rollout storage同一batch_idx的current observation/history。
  保持默认encoder minibatch接口不变，只在anchor启用时附加current observation。
  增加跨时刻/打乱batch索引的配对测试，四项anchor/diagnostic测试及1iteration smoke通过。
  v3的性能需独立重训验收，不能沿用v2的数值结论。

启动v3新实验（仓库根目录）：

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py --profile ENCODER_ANCHORED
```

该命令源固定为已验收model100，追加500到600，自动评估200/600三seed、每命令16env、
25秒/5秒warmup，命令0/±0.5/±1，结果在独立outputs目录。不会自动升速或替换基线。

本轮修改：`rsl_rl/algorithms/ppo.py`、`rsl_rl/storage/rollout_storage.py`、
`envs/wheel_legged/policy_experiments.py`、`scripts/train.py`（均在plane/wheel_legged_gym）、
`tools/run_h3_low_speed.py`；新增`plane/tests/test_encoder_anchor.py`及本文/结果JSON。
未修改机器人资产、PD、部署接口。最终±4m/s、±4rad/s、转弯/反向目标仍未完成。

## v3 结果与下一单变量对照

v3正式run：`Sep19_11-58-36_encoder_anchored_20260919_115829`，100→600完成；
200/600各三seed，共30条验收。model200通过3/15，model600通过11/15，仍未全过。
model600三seed均速为 `0:-0.0252、-.5:-0.4441、+.5:+0.4012、-1:-0.7975、+1:+0.9611`。
后退1m/s仍明显不足，+.5平均接近门槛但有seed未过。保持源model100，不升速。
证据见 `docs/data/encoder_anchor_v3_20260919.json`。

下一profile为 `ANCHORED_WHEEL_EXPLORE`，仍从同一model100完整续训，并使用v3 anchor。
仅把左轮std[2]初始化为源右轮std[5]（0.00125548→0.06185795），其他网络权重、
噪声通道、Adam状态、reward/采样/LR保持。这个约49倍差异可能限制左轮探索，
但是否是跟踪停滞原因尚未证实。部署仍是actor mean，不加命令/速度补偿。
64env×1iteration smoke与5项回归测试通过，manifest记录调整前后向量。

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py --profile ANCHORED_WHEEL_EXPLORE
```

它创建新run追加500iteration，200/600两点同条件验收，不替换旧模型。
须以新任务status/acceptance判断结果，不复用anchor v3结论。
