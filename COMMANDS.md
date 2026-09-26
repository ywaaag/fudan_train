# Fudan 轮腿训练仓库命令大全

本仓库位置：`/home/kellen/fudan_train`。训练使用 Isaac Gym Preview 4，
不要在 Isaac Lab/Isaac Sim 环境中运行这些命令。原始参考仓库
`/home/kellen/fudan_rl_wheel_leg/plane` 不要修改。

## 1. 每次启动前

```bash
cd /home/kellen/fudan_train/plane
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/kellen/fudan_train/plane
export CUDA_VISIBLE_DEVICES=0
```

如果首次使用该副本，安装为 editable package：

```bash
python -m pip install --no-deps -e .
```

如果出现 `libpython3.8.so.1.0` 找不到，确认已经执行上面的
`LD_LIBRARY_PATH` 命令；不要重新安装 Isaac Gym。

## 2. 检查环境和资产

```bash
python - <<'PY'
import torch, isaacgym
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
PY

python -m py_compile wheel_legged_gym/scripts/train.py wheel_legged_gym/scripts/play.py
python -m pytest -q wheel_legged_gym/tests  # 若环境没有 pytest，可跳过
```

重新生成树模型资产（需要 `mujoco` Python 包）：

```bash
cd /home/kellen/fudan_train
python tools/build_tree_urdf.py --help
python tools/build_parity_mjcf.py
```

## 3. 冒烟训练

```bash
cd /home/kellen/fudan_train/plane
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless \
  --num_envs=64 --max_iterations=10 \
  --seed=7 --run_name=self_urdf_smoke
```

## 4. 正式训练

无 GUI，适合长时间训练：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless \
  --num_envs=4096 --max_iterations=5000 \
  --seed=11 --run_name=baseline_v1
```

带 GUI（降低环境数）：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged \
  --num_envs=1 --max_iterations=1000 \
  --seed=11 --run_name=gui_debug_v1
```

可用 policy experiment：`H3 H7`。其中 `H3` 是历史低速恢复基线，`H7` 是当前唯一推荐的
固定 3.0 m/s 稳定化实验，使用 zero/small/反向/正向各 25% 和显式 ±3.0 m/s 锚点。
示例：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=4096 \
  --max_iterations=5000 --policy_experiment=H7 \
  --run_name=H7_fixed_3ms_v1
```

每次实验使用新的 `--run_name`，不要覆盖旧 run。

## 5. 从 checkpoint 继续训练

`play.py` 不能训练；续训必须用 `train.py --resume`：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=4096 \
  --resume \
  --load_run=Aug31_15-25-37_policy_zero_reverse_H_seed11 \
  --checkpoint=3800 \
  --max_iterations=1200 \
  --policy_experiment=H7 \
  --run_name=H7_fixed_3ms_v1
```

这表示从 `model_3800.pt` 加载，再训练 1200 个 PPO iteration，目标约为 5000。

查看 checkpoint：

```bash
find logs/wheel_legged -maxdepth 2 -name 'model_*.pt' -printf '%h/%f\n' | sort
```

## 6. 播放和 GUI 验收

```bash
python wheel_legged_gym/scripts/play.py \
  --task=wheel_legged \
  --experiment_name=wheel_legged \
  --load_run=Sep07_16-10-41_method_v1_translate_20_long_v1 \
  --checkpoint=5000
```

旧 run 示例：

```bash
python wheel_legged_gym/scripts/play.py \
  --task=wheel_legged --experiment_name=wheel_legged \
  --load_run=Aug31_15-25-37_policy_zero_reverse_H_seed11 \
  --checkpoint=3800
```

验收时观察：站立高度、roll、左右腿几何姿态、左右轮是否接触、前进/后退/转向方向。
只看 mean reward 不能证明策略正确。

## 7. TensorBoard

另开一个终端：

```bash
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
tensorboard \
  --logdir=/home/kellen/fudan_train/plane/logs/wheel_legged \
  --port=6006 --bind_all --reload_interval=5
```

浏览器打开：`http://127.0.0.1:6006`。

只查看单次实验：

```bash
tensorboard \
  --logdir=/home/kellen/fudan_train/plane/logs/wheel_legged/H_finetune_v1 \
  --port=6007
```

重点曲线：`Train/mean_reward`、`Train/mean_episode_length`、
`Policy/mean_noise_std`、`Loss/encoder`、`Policy/mean_kl`，以及日志中的
`reverse_tracking_abs_error`、`forward_tracking_abs_error`。

## 8. 导出 ONNX

```bash
python export_onnx/export_onnx.py \
  --load_run=H_finetune_v1 \
  --checkpoint=5000 \
  --out=outputs/H_finetune_v1_5000.onnx
```

如果只需要常规导出脚本，也可查看：

```bash
python export_onnx/export_onnx.py --help
python export_onnx/verify_onnx.py --help
```

导出文件通常位于 `outputs/`。导出后必须做 PyTorch/ONNX 数值一致性检查，不能只看
文件是否生成。

## 9. Isaac command-grid 评估

```bash
python wheel_legged_gym/scripts/isaac_command_grid.py \
  --checkpoint=logs/wheel_legged/H_finetune_v1/model_5000.pt \
  --seconds=10 \
  --out=outputs/H_finetune_v1_isaac_grid.json
```

以命令网格结果为准判断正向、反向、零速和 yaw，不要用单个 GUI 片段代替评估。

## 10. MuJoCo sim2sim 边界

训练仓库只负责 Isaac Gym tree URDF policy。完整闭链 MuJoCo 验证属于独立仓库：

```bash
cd /home/kellen/wheel_leg_sim2sim
python sim2sim_closed_policy.py --help
```

先在 Isaac Gym 通过 checkpoint 验收，再导出 ONNX，最后在独立 sim2sim 仓库中验证。

## 11. 停止进程

## 12. method_v1 分阶段训练

`method_v1` 保持 25D observation、5 帧 history 和 6D action contract。旧 H3/H7
仍用于历史 checkpoint 复现；新训练使用显式 phase/command level：

```bash
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/kellen/fudan_train/plane
export CUDA_VISIBLE_DEVICES=0
cd /home/kellen/fudan_train/plane

python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=4096 \
  --policy_experiment=method_v1 --phase=stand --command_level=0 \
  --max_iterations=5000 --run_name=method_v1_stand_v1
```

阶段顺序为 `stand`、`translate`、`yaw`、`combined`；translate/yaw 的
`command_level` 依次对应 `0.5/1/2/3/4`。从旧 checkpoint 迁移时只加载
actor/encoder，不加载旧 critic、optimizer 或课程状态：

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=4096 \
  --resume --resume_mode=policy \
  --load_run=Sep06_17-58-00_H7_unclipped_global_slip_from16600_v2 \
  --checkpoint=17600 --policy_experiment=method_v1 \
  --phase=stand --command_level=0 \
  --max_iterations=5000 --run_name=method_v1_from_h7_v1
```

先进行 64-env、1 iteration smoke，再进行 100--300 iteration probe；只有
zero/translation/yaw/contact/slip 指标稳定后才延长训练。`method_v1` 不允许未显式
指定 `--resume_mode=policy` 时加载旧 reward 语义的完整 checkpoint。

训练终端使用 `Ctrl+C`。TensorBoard 是独立进程；查看 PID：

```bash
ps -ef | grep '[t]ensorboard'
```

只停止 TensorBoard：

```bash
pkill -f '/fudan_leg/bin/tensorboard'
```

不要删除 `logs/`，其中包含 checkpoint、TensorBoard event 和每次运行保存的配置。

## 固定高度转弯包络（已完成的有界 A/B；不可原名重跑）

先读 `docs/modules/turn_envelope.md`，再读
`plane/outputs/turn_envelope_20260925_162526/README.md`。本轮 A/B 已完成，
下面命令只用于核对公开入口的参数形状。新实验必须另建输出目录、spec 和
唯一 `run_name`，且需要新的明确训练授权；不要覆盖本轮产物。

```bash
cd /home/kellen/fudan_train/plane
env PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
CUDA_VISIBLE_DEVICES=0 \
FUDAN_TURN_SPEC=/absolute/path/to/new_spec.json \
/home/kellen/anaconda3/envs/fudan_leg/bin/python \
wheel_legged_gym/scripts/train.py \
--task=wheel_legged --headless --num_envs=64 --max_iterations=1 --seed=23 \
--policy_experiment=TURN_ENVELOPE --resume --resume_mode=full \
--load_run=Sep22_11-19-04_motion_goal_20260922_111611_r01_basic_motion \
--checkpoint=10200 --run_name=NEW_UNIQUE_TURN_SMOKE
```

`new_spec.json` 的 schema 见本轮 `A/spec.json` 或 `B/spec.json`；只能选分支
`A`/`B`，来源 SHA 固定。64 环境 smoke 不能代替正式评估。执行评估时使用
`scripts/evaluate_policy_comparison.py --help` 核对 `--yaw-delay-seconds`、
`--yaw-exit-at` 和 `--yaw-exit-factor`；现有结果采用 16 环境/命令、
seed 19/37/53，原 25 命令与转弯网格分开验收。只读汇总入口：

```bash
cd /home/kellen/fudan_train
env PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
/home/kellen/anaconda3/envs/fudan_leg/bin/python \
tools/summarize_turn_envelope.py --help
```

## TURN_LEAN_LONG（本轮有界训练；原 run_name 禁止重用）

先读 `docs/modules/turn_lean_long.md` 与
`plane/outputs/turn_lean_long_20260925_175224/README.md`。以下是 2026-09-25
历史 stage1 的实际命令形状；该 run 已结束，不能再次执行相同名称或套用旧预算。
当前 2026-09-26 高度技能实验以其独立 `spec_stage*.json`、`protocol*.md`
和 job `status.json` 为准；新阶段须使用新 run_name 与精确来源 checkpoint/SHA。

```bash
cd /home/kellen/fudan_train/plane
env PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
CUDA_VISIBLE_DEVICES=0 \
FUDAN_TURN_LONG_SPEC=/home/kellen/fudan_train/plane/outputs/turn_lean_long_20260925_175224/spec_stage1.json \
/home/kellen/anaconda3/envs/fudan_leg/bin/python wheel_legged_gym/scripts/train.py \
--task=wheel_legged --headless --num_envs=4096 --max_iterations=10000 \
--seed=23 --policy_experiment=TURN_LEAN_LONG --resume --resume_mode=full \
--load_run=Sep22_11-19-04_motion_goal_20260922_111611_r01_basic_motion \
--checkpoint=10200 --run_name=turn_lean_long_20260925_stage1
```

只读训练摘要（不要先打开完整 stdout）：

```bash
cd /home/kellen/fudan_train
env PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
/home/kellen/anaconda3/envs/fudan_leg/bin/python \
tools/summarize_training.py \
/home/kellen/fudan_train/plane/logs/wheel_legged/Sep25_18-00-59_turn_lean_long_20260925_stage1 \
--window 50
```

`status.json`、`summaries/`、`probes/` 在本轮输出目录；阶段/最终评估
入口为 `tools/review_turn_lean_long.py --help`，汇总入口为
`tools/summarize_turn_envelope.py --long --job <phase_review目录>`。这些
命令会启动Isaac评估或写新文件，架构审查不要执行。
