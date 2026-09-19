# Encoder 冻结/更新受控对照（2026-09-19）

目的：解释速度分档跟踪差异，并验证独立更新encoder是否破坏actor已学能力。
此前真实速度替换仅是诊断，不作为训练或部署中的隐藏补偿。

## 固定协议

两组从同一已验收源完整续训：
`Sep19_09-17-38_h3_low_speed_20260919_091730/model_100.pt`。
profile：`ENCODER_FROZEN` / `ENCODER_UPDATING`。
完整继承actor/critic/encoder/std、PPO Adam和encoder Adam，源SHA与manifest校验保留。
训练seed23，4096 env，GPU0，每组追加500 iteration到600，初始iteration100。
固定H3_SPEED1 reward/采样/随机化level1、无推扰、LR1e-5、entropy.001、mirror.01。
仅冻结组跳过encoder optimizer.step；保留同样的batch生成、forward/backward与随机数消费，
不重置噪声，不同时改reward或critic。encoder的Adam状态在冻结组也必须保持原样。

两组各评估200/600，每点seeds19/37/53、16env/命令、25秒/5秒warmup，
命令0/±0.5/±1。全部15项独立门槛通过才算阶段成功，不能以平均值掩盖失败。
这是一个训练seed的配对消融，不能声称多训练seed统计显著性。

## 新诊断指标

每iteration写入run内`command_diagnostics.jsonl`，同时写TensorBoard的`Command/`：

- 七个实际命令分别统计sample count、env-seconds、实际占比。
- tracking/yaw/height在physics之后、reset之前统计，包括早死步；不混用reset后的状态。
- encoder bias/MAE与其推理时刻的pre-action真实速度配对。
- reward贡献取实际episode_sums增量，包含权重/dt/gate，不重复调用reward函数。
- 没有样本的bin记null且不写误导性的0误差。
- 各window按实际样本数聚合，不等权平均不同reset batch的均值。
- failure与timeout分开；训练正常周期timeout不当作跌倒。

记录encoder单独更新前后的固定256条observation/history动作差异及
`0.5 * sum((delta_mu/std)^2)`的均值。它表示同std高斯策略的均值变化所对应的KL，
不是完整rollout KL，也不是actor PPO已记录的KL。probe采自已存在的minibatch，不增加随机采样。

代码仅在这两个profile显式启用诊断，旧Episode日志保留作对照。

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_encoder_ablation.py
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/summarize_command_diagnostics.py \
  plane/logs/wheel_legged/<run> --window=100 --out=<summary.json>
```

第一个命令启动新的配对实验，不是恢复。中断先核对status、进程和checkpoint，不能重复启动。

## Smoke与测试

两组64env×1iteration smoke通过，15项回归测试通过。
测试首次因utils/envs循环导入收集失败，按现有项目初始化顺序修正测试import后通过；
没有修改运行时包导入机制。

逐tensor验证：冻结组encoder与源完全一致；更新组encoder改变；两组首次PPO更新后的
所有非encoder模型tensor完全一致。3072个rollout样本全部计入诊断。
冻结组encoder-induced action shift为0；更新组smoke的左右轮mean abs action shift
约0.0276/0.0291。左轮std约0.0013，该probe均值变化对应KL约363.8。
这只是小smoke的风险证据，不能直接代表正式4096env训练的数值或因果结论。

正式job：`plane/outputs/encoder_ablation_20260919_100936/`。
终态和实际run路径以其status.json为准。

## 已完成结果

两组均完成100→600，每组500个诊断window、98,304,000个训练样本。
12组独立验收、60个命令/seed条目完成；初始状态和配置与源model100验收完全一致。

| 分支 / iteration | 完整门槛通过条目 | 低速9项全部通过 | 失败重置次数 |
|---|---:|---|---:|
| frozen 200 | 8/15 | 否 | 0 |
| frozen 600 | 0/15 | 否 | 516 |
| updating 200 | 12/15 | 是 | 0 |
| updating 600 | 4/15 | 否 | 0 |

失败重置次数不是失败环境数，同一环境可多次重置。frozen600的速度均值包含重置轨迹，
只用于描述退化，不作为跟踪成功证据。

三seed平均实际vx：

| 命令 | frozen200 | updating200 | frozen600 | updating600 |
|---|---:|---:|---:|---:|
| 0 | +0.0063 | -0.0113 | +0.2076 | -0.0603 |
| -0.5 | -0.3827 | -0.4156 | -0.3350 | -0.5005 |
| +0.5 | +0.3992 | +0.4177 | +1.9145 | +0.3472 |
| -1 | -0.6934 | -0.7256 | -0.8229 | -0.8562 |
| +1 | +0.9799 | +0.9424 | +2.1041 | +0.8945 |

推荐模型不变，仍为此前低速通过的
`Sep19_09-17-38_h3_low_speed_20260919_091730/model_100.pt`。
本轮没有新ONNX导出、没有sim2sim、没有升速；两组训练进程均结束。

### 机制结论与证据边界

1. **直接冻结encoder不是修复方案。** 冻结组600的encoder权重与其Adam状态和源逐项
   完全一致，冻结确实生效；但新状态分布下速度估计失配，最后出现超速/失败。
   更新组没有失败，但仍未全面通过，所以“估计器需要适应”不等于“当前更新方法已足够好”。
2. **更新组不是明显缺少端点样本。** 全程实际占比：zero20.03%，±0.1各约10%，
   ±0.5/±1各约15%。冻结组+1的时长占比降到13.05%，同时出现大量失败；不能把名义
   15%采样比例当作失败后仍然准确的rollout覆盖率。
3. **encoder单独更新确实改变有效策略。** 更新组末100次，轮子mean abs action shift
   左/右约0.02505/0.02050，probe implied KL均值约124.50；冻结组为0。
   actor PPO自身的clip/KL不覆盖之后这个独立encoder更新。小std使此KL特别敏感。
   这是已测量的优化机制风险；本次freeze对照没有证明简单消除该变化就会提升跟踪。
4. **准确估计不等于正确控制。** 更新组末100次的-1训练MAE约0.211，encoder bias
   接近0，MAE约0.03；因此-1不足不能全部归因于速度估计错误。
5. **reward记录没有显示所有误差都由巨大力矩/姿态惩罚压制。** 更新组-1末窗口，
   coarse/fine速度奖励约+0.715/+0.229，yaw奖励约+0.425/+0.213，height/slip成本
   约-0.073/-0.051，gap成本约-0.006（均为每env-second的实际加权贡献）。
   跟踪不准仍可获得较多总奖励；但这些贡献不是梯度或反事实最优值，不能据此断言
   哪个reward scale就是根因，更不能直接再放大tracking权重。

更新组model200/model600与上一轮未增加诊断的H3_SPEED1对应checkpoint：
**所有网络tensor逐项完全一致（max_abs_delta=0）**。诊断代码没有改变该对照分支训练结果。
原始验证在任务`instrumentation_check.json`。

### 下一步建议

保留encoder更新，但对它造成的策略变化施加可检查的约束：下一次单变量实验可比较
当前encoder更新与“同一batch上限制encoder更新后动作分布偏移”的版本，先验证
约束生效与估计精度，再做500iteration配对验收。暂不同时改reward、噪声或采样。
该策略尚未实现/验证，本次不宣称它一定有效。

完整的manifest、哈希、逐命令验收、按样本加权统计和机制校验见
[精简证据](data/encoder_ablation_20260919.json)。只有训练seed23，结论仅适用于
该已核查起点和配对设置；不能声称所有训练seed或所有模型都如此。

新增结果文件：`docs/data/encoder_ablation_20260919.json`；同步更新README及模型索引。

## 文件范围

新增：`plane/wheel_legged_gym/utils/command_diagnostics.py`、
`plane/tests/test_command_diagnostics.py`、`tools/run_encoder_ablation.py`、
`tools/summarize_command_diagnostics.py`及本文档。
增量修改：`plane/wheel_legged_gym/rsl_rl/runners/on_policy_runner.py`、
`plane/wheel_legged_gym/rsl_rl/algorithms/ppo.py`、
`plane/wheel_legged_gym/envs/wheel_legged/policy_experiments.py`、
`plane/wheel_legged_gym/scripts/train.py`。已有用户改动/模型全部保留。
