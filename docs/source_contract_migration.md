# `fudan_rl_wheel_leg` 移植契约对照

这份项目来自只读参考仓库 `/home/kellen/fudan_rl_wheel_leg`。移植并非行为等价复制：
当前项目替换了 asset、初始化、PD、reward、command sampler、termination 和部分 PPO
训练逻辑。此前 H3/method_v1 结果必须解释为当前 tree asset 下的新实验。

## 已确认的主要变化

| 区域 | 原项目 | 当前项目 | 影响 |
|---|---|---|---|
| asset | `infantry_V4_increase.urdf` | `assets/wheel_leg_train.urdf` | 质量、惯量、连杆、接触和关节命名不同 |
| reset | base z=`0.1`；腿角 `+0.2/+0.4`、`-0.2/-0.4` | base z=`0.4`；腿角全零 | 平衡点和初始姿态不同 |
| wheel PD | damping=`0.2` | damping=`1.0` | wheel velocity 对 torque 的响应不同 |
| reward | `exp(-error² / tracking_sigma)`，`sigma=.25` | normalized coarse/fine/Huber gap、gate、slip、接触和 zero reward | reward 数值及梯度语义不同 |
| command | 原始 resample/uniform | method_v1 固定 episode command、分段 counter、anchors | 命令覆盖与 credit assignment 不同 |
| termination | 原始 contact/姿态逻辑 | contact history、grace、upright streak、wheel loss | episode length 和失败样本不同 |
| PPO | 默认 LR `1e-3`、entropy `.01` | 当前迁移 profile LR `1e-5`、entropy `.001` | 更新速度相差两个数量级 |
| encoder | extra optimizer | extra optimizer 加镜像 estimator loss；之后独立更新 encoder | actor 有效输入可在 PPO 后继续变化 |

逐文件 hash、AST literal 配置变化和差异摘要见
[source_contract_comparison_20260919.json](data/source_contract_comparison_20260919.json)。

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/compare_source_contract.py
```

该工具只读参考仓库，不修改参考目录，也不启动训练。

## 当前结论

当前速度跟踪变差不能直接归因于某一个 reward 或 encoder bug。首先存在一个更基础的
移植问题：**训练对象已经不是原项目的同一个物理/训练契约。** 因此从原项目经验直接
迁移 checkpoint、课程和 reward 组合，会同时改变多个变量，无法形成因果对照。

当前低速 model100 仍保留为新 tree asset 下的有效基线；原项目参考目录保持只读。
后续若要复现原项目，应建立明确的 `legacy_parity` profile，逐项恢复原 asset、reset、
PD、reward、command 和 PPO，再用短 smoke 验证。不能在当前 tree asset 上宣称“原项目
等价复现”，也不能用 command offset、真实速度替换或隐藏补偿伪造等价性。

本次只完成契约审计和报告，没有改变训练 profile、checkpoint 或参考仓库。
