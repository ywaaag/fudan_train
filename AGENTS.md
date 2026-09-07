# Fudan 轮腿机器人训练仓库协作说明

## 1. 项目身份与目录边界

本文件适用于独立仓库：

```text
/home/kellen/fudan_train
```

这是一个基于 Isaac Gym Preview 4 的轮腿机器人强化学习训练仓库，主要负责：

- 训练 six-DOF tree approximation policy；
- 保存和评估 PPO checkpoint；
- 导出 ONNX；
- 为后续 MuJoCo sim2sim 提供策略。

不要把以下目录混为同一个项目：

```text
/home/kellen/fudan_rl_wheel_leg/plane
```

这是 Fudan 原始参考仓库，禁止修改。

```text
/home/kellen/wheel_leg_mjrl-lqr
```

这是旧的综合工作区和 sim2sim 参考目录，保留其中已有改动，不要用它覆盖本仓库。

```text
/home/kellen/wheel_leg_sim2sim
```

这是独立的完整闭链 MuJoCo 验证仓库。训练仓库不要重新实现或临时修改 MuJoCo
闭链控制器来掩盖训练策略问题。

本仓库当前是新迁移出来的独立 Git 仓库，默认分支为 `main`。仓库尚未假设有
远程 origin，也不要未经用户要求执行 push、清理日志或删除 checkpoint。

## 2. 物理和策略边界

训练使用：

- `assets/wheel_leg_train.urdf`：六个 scalar DOF 的树模型近似；
- `assets/wheel_leg_train_parity.xml`：由 URDF 生成的 MuJoCo parity tree；
- `wheel_legged_gym/`：Isaac Gym 环境和 Fudan 风格 PPO；
- `meshes_mj/`：URDF visual mesh 资源；
- `wheeled_infantry.xml`：完整闭链模型，仅作为资产生成和独立 sim2sim 参考。

固定 policy contract，不要随意改变：

```text
observation: 25D
history: 5 frames = 125D，oldest -> newest
action: 6D，left-first
action order:
  [left_leg_0_position,
   left_leg_1_position,
   left_wheel_velocity,
   right_leg_0_position,
   right_leg_1_position,
   right_wheel_velocity]
physics dt: 0.005 s，Isaac Gym 默认 200 Hz
policy decimation: 2，policy 100 Hz
position action scale: 0.5
wheel velocity action scale: 10.0
Kp: [20, 20, 0, 20, 20, 0]
Kd: [1, 1, 1, 1, 1, 1]
torque limits: [40, 40, 47.294118, 40, 40, 47.294118]
```

`ActorCriticSequence` 的网络结构为：

- encoder：`125 -> 128 -> 64 -> 3`；
- actor：`25 + 3 -> 128 -> 64 -> 32 -> 6`；
- critic：训练时可接 privileged observation 和 latent，不能把 critic 输入误当成部署输入。

动作不是直接 torque policy。腿部 action 是 position target，轮子 action 是
velocity target，最终由混合 PD 生成关节 torque。不要把这个接口改成端到端 torque，
除非用户明确重新冻结一套新的 contract。

## 3. 环境启动

必须使用旧版 Isaac Gym 环境：

```bash
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/kellen/fudan_train/plane
export CUDA_VISIBLE_DEVICES=0
cd /home/kellen/fudan_train/plane
```

当前已知运行时：

- Python 3.8；
- PyTorch 2.4.1 + CUDA 11.8；
- Isaac Gym Preview 4；
- 单卡默认使用 GPU 0（RTX 4080）。

不要在 `issaclab`、`isaac` 或 Isaac Sim 新版本环境中运行这里的
`wheel_legged_gym/scripts/train.py`。如果出现：

```text
libpython3.8.so.1.0: cannot open shared object file
```

通常是没有设置 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:..."`，不是 Isaac Gym 安装损坏。

首次安装 editable package：

```bash
python -m pip install --no-deps -e .
```

不要用 pip 自动升级 Isaac Gym、PyTorch 或 CUDA 相关组件。

## 4. 训练入口和日志规则

训练入口：

```bash
python wheel_legged_gym/scripts/train.py
```

完整命令、TensorBoard、导出和评估示例见：

```text
/home/kellen/fudan_train/COMMANDS.md
```

每次实验必须：

1. 使用新的 `--run_name`；
2. 记录 seed、checkpoint、`policy_experiment` 和实际 reward 配置；
3. 先运行小规模 smoke；
4. 再运行正式训练；
5. 用 Isaac command-grid 验收；
6. 导出 ONNX 并做数值一致性检查；
7. 最后才进入独立 MuJoCo sim2sim。

不要删除 `logs/wheel_legged/`。其中包含：

- `model_*.pt`；
- TensorBoard event；
- 本次运行保存的 `wheel_legged_config.py`、`legged_robot.py`；
- `policy_experiment.json`；
- 复现实验所需的 optimizer 和课程信息。

## 5. 当前课程和实验配置

实验定义在：

```text
plane/wheel_legged_gym/envs/wheel_legged/policy_experiments.py
```

当前可用实验名称：

```text
A B C R L Y S T H H2 H3
```

其中：

- `A/B/C`：zero/reverse mixture 的早期实验；
- `L/Y`：低速 anchor 和低速 tracking 实验；
- `S/T`：旧的左右轮 action symmetry 实验，不建议继续使用；
- `H`：性能门控课程；
- `H2`：提高 zero/small 比例并加入 `±2.5/±3.0/±3.25/±3.5 m/s` 端点锚点；
- `H3`：H2 的保守微调版本，降低 learning rate 和 zero penalty，仍使用端点锚点。

课程阶段目前为：

```text
0.5 / 0.8
1.0 / 1.0
2.0 / 2.0
3.5 / 3.5
5.0 / 5.0
```

分别表示 linear speed limit 和 yaw rate limit。进入某个 stage 不等于通过该 stage。
只有 `curriculum_window_passed=1`，并且生存率、zero、forward、reverse、yaw 误差均达标，
才算真正通过。

命令采样实现位于：

```text
plane/wheel_legged_gym/envs/base/command_sampling.py
```

重要语义：

- zero mode 的 forward/yaw 都是精确 `0.0`；
- small mode 可使用离散 anchor；
- reverse mode 只采样非正线速度；
- forward mode 只采样非负线速度；
- endpoint anchor 会被裁剪到当前 curriculum range；
- endpoint anchor 只作用于 forward/reverse，不会破坏 zero/small mode。

## 6. 当前已知训练状态

最近一次有效 H2 run：

```text
logs/wheel_legged/Sep05_17-17-34_H2_anchor_from15600_v1
```

H2 在进入 3.5 m/s stage 后发生退化：

- `curriculum_linear_limit = 3.5`；
- `curriculum_window_passed = 0`；
- 最后记录的 survival fraction 约 `0.53`；
- forward relative error 约 `0.46`；
- reverse relative error 约 `0.41`；
- yaw relative error 约 `0.81`；
- zero abs vx 约 `0.075 m/s`；
- zero wheel speed 约 `3.84 rad/s`。

最近的 H3 保守微调 run：

```text
logs/wheel_legged/Sep05_17-41-43_H3_from_H2_best_v1
```

该 run 已完成约 3000 个追加 iteration，但最后记录仍未通过 3.5 stage：

- 最后 iteration 约 `18699`；
- `curriculum_window_passed = 0`；
- survival fraction 约 `0.486`；
- forward relative error 约 `0.473`；
- reverse relative error 约 `0.423`；
- yaw relative error 约 `0.761`；
- zero abs vx 约 `0.079 m/s`；
- zero wheel speed 约 `3.28 rad/s`。

因此不要把 H3 的最新 checkpoint 当作成功模型，也不要只因为 reward 上升就宣布
高速控制成功。优先选择有明确指标的最佳 checkpoint，而不是默认最新 checkpoint。

## 7. 对称性问题的正确处理

曾经的问题是左右腿长期不对称。已确认不能简单地提高 raw action symmetry reward。
原因包括：

- 左右物理关节轴、canonical sign 和 body frame 语义可能不同；
- 平衡过程中需要短时左右差动来纠正 roll、接触和质量中心误差；
- 强制 `left_action == right_action` 会把补偿转移到另一条腿，导致更严重的不对称；
- 训练树模型的 COM 本身存在小横向偏置，物理最优控制不一定是逐时刻完全对称。

当前已验证的 wheel-leg URDF 顺序为：

```text
left_leg_0, left_leg_1, left_wheel,
right_leg_0, right_leg_1, right_wheel
```

不要对所有右腿通道统一乘 `-1`。如果要增加 symmetry regularization，必须同时镜像：

- angular velocity；
- projected gravity；
- yaw command；
- 左右 leg position channels；
- 左右 wheel velocity channels；
- previous action；
- 五帧 history。

并通过 mirrored observation -> mirrored action 的 equivariance 约束实现，而不是直接
惩罚同一时刻左右 action 差值。

## 8. 高速训练的建议流程

用户目标是逐步达到：

- forward/backward 约 `4 m/s`；
- yaw spin 约 `4 rad/s`；
- zero command 时尽量接近零速度；
- 保持双轮接触、不底盘触地、不提前 termination。

自主迭代时遵循以下顺序：

1. 从已知最好 checkpoint 开始，不从退化末端盲目续训；
2. 每轮只改变一个主要变量（采样、reward、optimizer 或 curriculum）；
3. 先做 1 iteration smoke；
4. 再做短训练观察 100~300 iteration；
5. 如果 reward、episode length、survival 或 tracking 同时恶化，暂停并回退；
6. 不用 command offset、speed compensation、hidden LQR、static trim 或 command gate
   伪造成功；
7. 只有在低速、zero、reverse、yaw 都稳定后，才提升到 3.5/4.0；
8. 高速阶段需要显式 anchor，例如 `±3.0、±3.5、±4.0 m/s`，不能只依赖 uniform；
9. yaw 也应有 `±2、±3、±4 rad/s` 的显式 anchor 或分阶段训练；
10. 每个候选 checkpoint 通过 Isaac command-grid 后再导出 ONNX 和跑 sim2sim。

如果连续多轮没有改善，应先暂停训练并检查：

- `Episode/curriculum_window_passed` 是否一直为 0；
- `Episode/curriculum_survival_fraction` 是否低于门槛；
- forward/reverse/yaw relative error 是否下降；
- zero abs vx 和 zero wheel speed 是否下降；
- action noise std 是否失控；
- torque saturation 是否升高；
- 当前命令实际均值是否真的覆盖目标高速端点。

## 9. 训练、续训和查看命令

基础环境：

```bash
cd /home/kellen/fudan_train/plane
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/kellen/fudan_train/plane
export CUDA_VISIBLE_DEVICES=0
```

Smoke：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=64 \
  --max_iterations=10 --seed=11 --run_name=smoke_v1
```

H3 从 H2 最佳点续训：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=4096 \
  --resume \
  --load_run=Sep05_17-17-34_H2_anchor_from15600_v1 \
  --checkpoint=15700 \
  --policy_experiment=H3 \
  --max_iterations=3000 \
  --run_name=H3_from_H2_best_v2
```

GUI play：

```bash
python wheel_legged_gym/scripts/play.py \
  --task=wheel_legged \
  --experiment_name=wheel_legged \
  --load_run=<run_name> \
  --checkpoint=<iteration>
```

TensorBoard：

```bash
tensorboard \
  --logdir=/home/kellen/fudan_train/plane/logs/wheel_legged \
  --port=6006 --bind_all --reload_interval=5 --load_fast=false
```

浏览器：`http://127.0.0.1:6006`。

TensorBoard 兼容性：

- TensorBoard 当前使用 2.14.0；
- `fudan_leg` 中 protobuf 必须保持在 `4.25.3`；
- 如果出现 `Failed to fetch runs`，先检查：

```bash
python -c "import google.protobuf; print(google.protobuf.__version__)"
```

应输出 `4.25.3`。若不是：

```bash
python -m pip install --force-reinstall 'protobuf==4.25.3'
```

ONNX 导出：

```bash
python export_onnx/export_onnx.py \
  --load_run=<run_name> \
  --checkpoint=<iteration> \
  --out=outputs/<run_name>_<iteration>.onnx
```

PyTorch/ONNX 检查：

```bash
python export_onnx/verify_onnx.py \
  --checkpoint=logs/wheel_legged/<run_name>/model_<iteration>.pt \
  --onnx=outputs/<run_name>_<iteration>.onnx
```

Isaac command-grid：

```bash
python wheel_legged_gym/scripts/isaac_command_grid.py \
  --checkpoint=logs/wheel_legged/<run_name>/model_<iteration>.pt \
  --seconds=10 \
  --out=outputs/<run_name>_<iteration>_isaac_grid.json
```

## 10. 验收标准

单纯满足以下任一项都不能宣布成功：

- mean reward 上升；
- GUI 看起来没有立即倒下；
- episode length 较长；
- 某一个方向的速度看起来正确。

至少需要同时检查：

- zero command 的真实 `base_vx` 接近 0；
- forward 和 backward 方向正确且误差可接受；
- yaw `+/-` 方向正确；
- 目标高速附近有足够样本；
- survival fraction 足够高；
- `curriculum_window_passed=1`；
- 没有持续 torque saturation；
- 左右 wheel 接触稳定；
- policy checkpoint、ONNX 和 sim2sim 输入输出 contract 一致。

## 11. 修改规则

- 使用 `apply_patch` 修改代码；
- 修改前先查看当前 Git 状态和 run manifest；
- 不覆盖用户已有的 dirty changes；
- 不删除 logs、outputs、legacy checkpoints；
- 不修改 Fudan reference；
- 不把诊断用 gravity compensation、LQR、static trim 或 guard 混入正式 policy；
- 修改 reward 后必须做 smoke 和至少一次指标检查；
- 最终报告必须列出确切修改文件、运行命令和验证结果。

如果用户只要求诊断，禁止顺手修改训练代码。只有用户明确要求“调整训练”或
“实现目标”时，才修改课程、采样、reward 或 PPO。
