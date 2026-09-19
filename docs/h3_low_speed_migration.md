# H3 运动权重迁入低速课程（2026-09-19）

目的：保留统一验收中表现较好的 H3 运动权重，在 normalized_v1 框架内训练
±0.5 m/s。它是迁移可行性短训，不是对历史 LOW_SPEED 完整续训的单变量因果消融。

## 模型与迁移边界

源：`plane/logs/wheel_legged/Sep05_17-41-43_H3_from_H2_best_v1/model_15800.pt`。
新 profile：`H3_LOW_SPEED`。必须 `--resume --resume_mode=policy`，源 checkpoint
SHA256 必须匹配 `docs/data/policy_comparison_20260919.json` 的已评估 H3。

- actor/encoder 完整载入，并在首次训练前逐 tensor 精确检查。
- critic 重建；PPO 和 encoder optimizer 重建，iteration 从 0 开始。
- std 显式复制源模型，不沿用 policy-only 默认初始化的 0.5。
- 原有 LOW_SPEED 和 method_v1 的 full-resume 安全检查保留。
- URDF、PD、dt、25/125/6 接口不变。

源六通道 std：`[0.0897034, 0.2305856, 0.00127797, 0.0756774, 0.2405494, 0.0608320]`。
左右 wheel std 差异很大，作为本轮记录的风险，暂不擅自对称化或归因于它。

## 首轮设置

与 LOW_SPEED 保持同一 reward、采样和 optimizer 设置：20% zero、20% ±0.1、
30% reverse −0.5、30% forward +0.5，yaw=0，height=0.40。
随机化 level 1，无训练推扰；actor/critic、encoder LR=1e-5 fixed，entropy=0.001，
tracking_linear_cap=1.0，mirror loss=0.01，关闭站立几何/stand_still penalty。
seed=23，4096 env，500 iteration，GPU 0。

Smoke：64 env、1 iteration，run 为
`Sep19_09-15-57_h3_low_speed_migration_smoke_20260919/model_1.pt`，迁移断言及训练通过。
10 项迁移/低速/gate 测试通过；Isaac Gym 的 NumPy 废弃别名 warning 不影响执行。

自动任务（从仓库根目录，用 fudan_leg）：

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py
```

此命令启动一个新实验，不是恢复或查询。全局实验锁阻止两个该 supervisor 并行。
中断后先检查 status/进程/checkpoint，不重新执行此命令冒充续训。

本次任务：`plane/outputs/h3_low_speed_20260919_091730/`。
正式运行结束后，对 model_100 和 model_500 分别做三种子、每命令 16 env、
25秒/5秒 warmup 的统一 deterministic 验收。命令 0、±0.5 是当前课程门槛，
±1 是运动能力保留诊断，不表示允许升速。门槛与上一轮统一验收相同。
详细命令、run_dir、状态、原始结果在任务目录；最终性能需以验收为准。

## 已完成结果

正式 run：`Sep19_09-17-38_h3_low_speed_20260919_091730`。
500 iteration 正常结束；100/500 两点各完成三种子共 30 个命令/seed 条目。
两个 checkpoint 均通过当前 `0、±0.5` 门槛（各 9/9），但均未通过 ±1。
所有本次审计均无 failure/timeout，双轮接触 100%，无非轮触地，力矩饱和为 0。
实际初始状态、observation、配置与旧 H3 统一验收逐项相同。

| 命令 m/s | 源 H3 实际 vx | model_100 实际 vx | model_500 实际 vx |
|---|---:|---:|---:|
| 0 | -0.0068 | -0.0080 | -0.0362 |
| -0.5 | -0.4032 | -0.4087 | -0.4677 |
| +0.5 | +0.3799 | +0.4304 | +0.4258 |
| -1 | -0.8686 | -0.7999 | -0.7785 |
| +1 | +0.9456 | +1.2643 | +1.3279 |

选择 **model_100** 为当前低速候选：相较500，停车漂移较小、零命令高度 MAE
约3.9 mm而非20.8 mm、零命令轮心镜像误差约9.5 mm而非49.3 mm。
model_500 的后退误差更低（0.0323而非0.0913 m/s），仍保留为对照；不宣称100所有指标都更好。

```text
plane/logs/wheel_legged/Sep19_09-17-38_h3_low_speed_20260919_091730/model_100.pt
plane/outputs/h3_low_speed_20260919_091730/model_100.onnx
```

ONNX 已导出，256 组固定随机输入的 PyTorch/ONNX allclose 检查通过，日志见任务目录
`onnx_verify.log`。精确模型/ONNX哈希、迁移参数与30条验收数据保存于
[本轮证据](data/h3_low_speed_20260919.json)。导出一致性不是闭链 sim2sim 验收。

本轮未自动升速。下一阶段应从100继续受控扩大到±1，并保留0/±0.5回归门槛，
重点修正+1超调和-1不足。当前 H3_LOW_SPEED 限定原始H3迁移源，不能直接当作
model_100完整续训入口；下一阶段需要独立且具备manifest一致性检查的续训配置。

测试命令：

```bash
env PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python -m pytest -q \
 plane/tests/test_h3_low_speed.py plane/tests/test_low_speed.py \
 plane/tests/test_policy_comparison_gate.py
```

结果10 passed。没有改动旧 LOW_SPEED reward、物理资产和 policy contract；新增profile
仅定义本次显式迁移，保留所有已有checkpoint与未提交修改。

本轮新增文件：

- `plane/wheel_legged_gym/envs/wheel_legged/h3_low_speed.py`
- `plane/tests/test_h3_low_speed.py`
- `tools/run_h3_low_speed.py`
- `docs/h3_low_speed_migration.md`
- `docs/data/h3_low_speed_20260919.json`

本轮增量修改文件：

- `plane/wheel_legged_gym/envs/wheel_legged/policy_experiments.py`
- `plane/wheel_legged_gym/scripts/train.py`
- `README.md`
- `docs/model_branches.md`

完整训练/验收命令保存在任务目录的status.json及工具中；日志、checkpoint和ONNX
属于上述logs/outputs下的新产物，没有覆盖旧实验。
