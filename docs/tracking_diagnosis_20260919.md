# 速度跟踪偏差诊断（2026-09-19）

本次只诊断，不修改训练代码、reward、PPO或正式策略。
checkpoint：`Sep19_09-38-35_h3_speed1_20260919_093828/model_600.pt`。
三个条件均为seed19、每命令16env、25秒/5秒warmup、level1随机化。
baseline来自既有验收；另运行采样动作和真实速度替换两项临时进程内诊断。

| 命令 | 正常deterministic | 按checkpoint std采样动作 | actor输入改用真实三维base速度 |
|---|---:|---:|---:|
| 0 | -0.0618 | -0.0592 | -0.0113 |
| -0.5 | -0.5005 | -0.4971 | -0.4391 |
| +0.5 | +0.3434 | +0.3491 | +0.4477 |
| -1 | -0.8536 | -0.8529 | -0.8024 |
| +1 | +0.8883 | +0.8846 | +0.9864 |

数值为真实vx均值，m/s。三组均无失败。
真实速度替换使用与encoder相同的lin_vel缩放，保持actor权重、命令、物理不变；
替换的是三个latent速度通道，不是只替换vx，因此不能将作用完全归给vx估计。
原始JSON在 `plane/outputs/h3_speed1_20260919_093828/`：
`model600_seed19.json`、`diagnostic_sampled_seed19.json`、`diagnostic_true_velocity_seed19.json`。
后两者标记为diagnostic，不用于模型验收或部署。

## 证据支持的结论

- 这组实验不支持“训练采样/部署取均值”是主要原因；恢复动作采样后结果很接近。
- 用真实速度替换估计量明显改善零命令和前进，说明估计器与actor组合对误差有实质影响。
- 后退在替换后更差，说明actor不是一个可直接替换估计器即全面改善的控制器；
  存在方向相关的估计误差/已学习补偿。不能声称encoder是唯一根因。
- 原验收的接触率、slip、torque saturation不支持当前误差主要由打滑/力矩顶满造成。
- training metric存在汇总局限：reset_idx对无某方向样本的batch产生0，runner对各batch
  均值等权平均；且forward/reverse混合不同速度，不能代替每端点MAE。
  这会在无样本batch存在时稀释误差；本次没有量化历史每个batch的稀释比例。
- PPO先更新actor/critic/std，之后用速度监督loss独立更新encoder；actor用detached latent。
  因此encoder更新可改变actor输入，且不受同一次actor PPO ratio clipping约束。
  这是源码确认的机制风险，不是已完成冻结encoder消融后的因果结论。

下一步优先建立按实际命令/time-step加权的统计，再在同一已验收源上做冻结encoder与
继续更新encoder的单变量对照，保留0/±0.5/±1全部门槛；同时记录每方向encoder bias/MAE。
不能把真实速度替换、command offset或其他隐藏补偿混入部署以宣称成功。
仅一个诊断seed，尚不能推广为所有checkpoint/seed的定量因果解释。
