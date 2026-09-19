# Fudan 轮腿机器人训练仓库协作说明

## 0. 当前交接快照（2026-09-19，优先阅读）

本节更新旧 H 系列记录；冲突时优先核对本节、当前源码和 run manifest。
路径、Git 状态和进程必须现场验证，不要把历史快照当成运行中状态。

### 用户目标和工作方式

- 最终目标：前后 ±4 m/s、原地自转 ±4 rad/s、自然转弯、速度命令响应顺滑。
- 当前先做低速课程。零命令平稳停车仍需保留，但不优先训练世界位置锁定。
- 站立尽量双腿对称，运动允许必要的左右差动；不能硬锁 raw action 相等。
- 短训统一 **500 iteration**，smoke 可为 1 iteration，独立验收通过后才升速。
- 用脚本汇总 TensorBoard/event 和验收 JSON，不持续展开完整训练 stdout。
- 中断后检查进程、checkpoint 和 status.json 并继续，不能重复启动已有训练。
- 保留所有 logs/outputs/checkpoint 和用户 dirty changes；两个参考目录均只读：
  `/home/kellen/fudan_rl_wheel_leg/plane`、
  `/home/kellen/wheel_leg_mjrl-lqr/fudan_train/plane/wheel_legged_gym`。

### 已验证的站立基线

```text
plane/logs/wheel_legged/Sep18_21-21-37_stand_validated_20260918_212129/model_3100.pt
```

`STAND_SYMMETRIC` 保留原站立 reward，新增弱几何对称项 -0.1：比较机身坐标系内
膝点/轮心镜像距离，需 inverse rotation，容差 5 mm。固定 actor LR=1e-5，encoder
冻结，level 1 参数随机化，完整保留网络、std、optimizer 续训。

已做三种子（19/37/53）、每组 32 环境、60 秒小推扰验收。推扰为 0.10 m/s 水平
速度增量，不是标定力值，不能推广到强推扰或实机。仍有约 60 秒累计 1.1 m 慢漂移，
不能称为位置保持。用户后续明确优先运动能力。
详见 `docs/stand_balance_ablation.md`、`docs/standing_push_validation.md`。

对称性不能只看 q 差或 policy symmetry loss；同时检查实际几何、roll/pitch、接触和
失败率。当前 tree URDF 左右腿轴同向，不套用 full USD 右髋负号。
play 初始高度已改为 0.40 m，但仍保留部分随机化；严格对照用 evaluate_standing.py。

### 当前低速课程及迁移方式

独立配置 `LOW_SPEED`，基于 method_v1 translate；课程 **从 level 0 开始**：

```text
level 0/1/2/3/4 -> ±0.5/±1/±2/±3/±4 m/s
```

首阶段采样：20% exact zero、20% ±0.1 锚点、30% −0.5、30% +0.5；yaw=0、
height=0.40 m。尚未进入 yaw、combined 或命令切换课程。
运动阶段关闭 stand_bilateral_geometry 和 stand_still，保留镜像等变 loss=0.01。
level 1 参数随机化，无训练推扰。actor/encoder LR 均为 1e-5，fixed，entropy=0.001。

从站立 model_3100 完整 warm start actor/encoder/critic/std/Adam 状态。
**任务和奖励改变，旧 critic 只作为初始化，不是同语义精确续训。**
critic/optimizer 迁移影响尚未隔离验证；不要全局删除 resume 安全检查。

### 两轮低速实验均未通过，禁止升速

两轮均 seed=23、4096 env，从同一 model_3100 追加 500 iteration 至 model_3600。

| 实验 | run | 验收目录（相对 plane/outputs） |
|---|---|---|
| LOW_SPEED | Sep18_23-25-44_low_speed_05_20260918_232537 | low_speed_05_20260918_232537 |
| LOW_SPEED_TRACKING | Sep18_23-49-14_low_speed_05_20260918_234907 | low_speed_05_20260918_234907 |

第二轮只把 tracking_linear_cap 从 1.0 改为 0.5 m/s，同时改变 coarse/fine/gap
误差归一化；其余采样、optimizer 和随机化相同。记录在 manifest.reward_parameters。

三种子实际平均 vx（m/s）：

| 命令 | LOW_SPEED | LOW_SPEED_TRACKING |
|---|---:|---:|
| −0.5 | −0.082 | −0.015 |
| 0 | +0.133 | +0.205 |
| +0.5 | +0.282 | +0.370 |

前进时绝对 yaw 均值约 0.162 / 0.284 rad/s。两轮 9/9 验收完成，gate_passed=false、
paused_on_regression。无跌倒、无非轮触地、双轮 contact=100%，但后退不足、停车前漂
和非期望转向不合格。保留站立基线，不用失败模型替换。下一次仍需检查 live processes。

### 诊断证据和下一步

- 验收逐步断言实际命令没有被采样器覆盖；命令进入当前 observation 的对应通道。
- 第一轮补测（seed19、8 env、15秒/5秒 warmup）：encoder vx MAE 后退约 0.0155、
  前进约 0.0196 m/s，明显小于 tracking error，暂不足以认定 encoder 是主因。
- 下一步先核对各命令实际 rollout 覆盖时长、早死/episode 汇总偏差及 reward 贡献。
- 再受控对照 critic/optimizer 迁移方式，不同时改奖励、采样、噪声和 optimizer。
- sampled action 与 deterministic inference 差异曾被过早称为根因，**因果证据不足**。
- 不盲目延长失败 run，不继续单纯加大 tracking 权重，不引入 command offset/隐藏补偿。

低速门槛：三种子 × (-0.5/0/+0.5)，每项 16 env、25秒/5秒 warmup、deterministic。
要求无 failure/timeout/非轮触地，各轮 contact≥0.99，平均高度误差≤0.03 m，
运动 vx MAE≤0.10、零命令≤0.05 m/s，绝对 yaw 均值≤0.10 rad/s。
尚未完成响应延迟、超调、加减速、反向切换或转弯顺滑性验收。

### 关键文件和操作入口

```text
plane/wheel_legged_gym/envs/wheel_legged/low_speed.py
plane/wheel_legged_gym/envs/wheel_legged/stand_balance.py
plane/wheel_legged_gym/envs/wheel_legged/policy_experiments.py
plane/wheel_legged_gym/scripts/train.py
plane/wheel_legged_gym/scripts/evaluate_standing.py
tools/run_low_speed.py
tools/summarize_training.py
docs/low_speed_course.md
```

从仓库根目录读取精简日志（不启动训练）：

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/summarize_training.py \
  plane/logs/wheel_legged/<run> --window=100
```

`tools/run_low_speed.py --profile LOW_SPEED_TRACKING` 会从站立 3100 **重新启动新实验**，
不是恢复已有 run 或只读查询。验收入口支持 `--profile LOW_SPEED --vx=-0.5`，必须传
checkpoint/out。完整命令和迁移限制见 `docs/low_speed_course.md`。

`training_completion_hook.py` 通过 codex exec 生成独立复盘报告，不向当前聊天注入
消息，不自动重启训练。它仅监听明确结束状态，不能判断未写终态的 supervisor 崩溃。

Git 基线：696f487（站立/工具）、b3d291b（首轮低速）。2026-09-19 快照中第二轮
tracking 改动未提交，README 另有保留的 dirty change；以后必须以 live git status 为准。

FUDAN_STAND 复刻实验曾学成约 0.08 m 的对称趴地姿态，不是有效站立模型，见
`docs/fudan_stand_experiment.md`。早期几何奖励失败实验见 `docs/standing_geometry_reward.md`。

### 不要重复的解释错误

- 轮半径 0.06 m，7.7–8.5 rad/s 对应 0.462–0.510 m/s，±0.5 运动下不是异常轮速。
- timeout 与跌倒分开统计，一次周期重置不能直接判失败。
- 参数随机化 level 与课程 command_level 是不同概念。
- 新阶段是手动升阶，旧 H 系列 curriculum_window_passed 不是所有配置的通用门槛。
- stand entropy=0.001 曾被 manifest 的 0.005 覆盖，已修正并有回归测试。

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

## 5. 历史 H 系列课程和实验配置（当前配置见第 0 节）

实验定义在：

```text
plane/wheel_legged_gym/envs/wheel_legged/policy_experiments.py
```

历史实验名称（不代表当前 CLI 可用清单）：

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

## 6. 历史 H2/H3 训练状态（非当前主线）

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
