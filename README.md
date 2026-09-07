# 自有机器人 Fudan Isaac Gym 独立训练仓库

这个仓库是从已验证的 `/home/kellen/fudan_rl_wheel_leg/plane` 独立复制的
第一阶段训练线。原 Fudan 目录保持不变，便于随时复现参考结果。

  cd /home/kellen/fudan_train/plane

  source /home/kellen/anaconda3/etc/profile.d/conda.sh
  conda activate fudan_leg

  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH=/home/kellen/fudan_train/plane
  export CUDA_VISIBLE_DEVICES=0

  python wheel_legged_gym/scripts/train.py \
    --task=wheel_legged \
    --headless \
    --num_envs=4096 \
    --resume \
    --load_run=Sep05_16-07-37_v4 \
    --checkpoint=15600 \
    --policy_experiment=H7 \
    --max_iterations=3000 \
    --run_name=H7_fixed_3ms_v1



  source /home/kellen/anaconda3/etc/profile.d/conda.sh
  conda activate fudan_leg

  tensorboard \
    --logdir=/home/kellen/fudan_train/plane/logs/wheel_legged \
    --port=6006 \
    --bind_all \
    --reload_interval=5 \
    --load_fast=false


## Asset

训练资产是 `assets/wheel_leg_train.urdf`：

- 6 个 scalar DOF；
- 4 个腿 position target，2 个轮 velocity target；
- 25D observation，5 帧 history（125D）；
- 质量总和与 `wheeled_infantry.xml` 的 compiled MJCF 一致；
- 被动五连杆和导向轮固定在名义姿态；
- 完整闭链模型仍使用项目根目录的 `wheeled_infantry.xml`，只用于后续 MuJoCo sim2sim。

该 URDF 是训练近似模型，不是最终闭链动力学模型。完整命令请见
[`COMMANDS.md`](COMMANDS.md)。

## Environment

使用旧版独立环境，不要在 Isaac Lab 环境中运行：

```bash
conda activate fudan_leg
export LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/kellen/isaacgym/isaacgym/python
cd /home/kellen/fudan_train/plane
python -m pip install --no-deps -e .
```

## Smoke training

```bash
mkdir -p logs/wheel_legged
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=64 \
  --max_iterations=10 --seed=7 --run_name=self_urdf_smoke
```

本次已验证通过。每次训练会按时间戳建立 run 目录，可用下面的命令找到最新 checkpoint：

```bash
find logs/wheel_legged -maxdepth 2 -name 'model_10.pt' -printf '%h/%f\n' | sort
```

这只证明资产加载、reset、rollout、PPO update 和 checkpoint 正常，不能视为策略已经学会，也不能用于实机。

## 可视化检查

将 `<run_name>` 替换为 `find` 输出的目录名（例如
`Aug29_21-31-42_self_urdf_final_smoke`）：

```bash
python wheel_legged_gym/scripts/play.py \
  --task=wheel_legged --experiment_name=wheel_legged \
  --load_run=<run_name> --checkpoint=10
```

## Next run

先用 `logs/wheel_legged/.../model_10.pt` 做有限 play/可视化检查，再决定是否增加到 100～500 iterations。每次只改一个变量；不要同时改资产、PD、reward 和 PPO。

训练前必须确认：默认姿态、左右对称、轮子接地、六个关节动作方向和关节极限。当前树状 URDF 的轮半径约为 `0.06 m`，默认零关节姿态下轮心相对 base 约为 `-0.33993 m`，因此目标车高保持 `0.40 m`，避免轮子穿入地面后被接触求解器锁死。若要研究 `0.30 m`，必须先定义对应的腿关节默认角和可行工作空间。正式 sim2sim 前还要在完整 MJCF 中实现闭链约束误差和虚拟串联力矩映射验证。

## Source

参考实现来源：Fudan `fudan_rl_wheel_leg`，commit
`8204e853dfd2ed06d85a322e1a998c3d20a3be2c`。本副本的自有资产生成器为
`tools/build_tree_urdf.py`。

## MuJoCo sim2sim

The independent MuJoCo validation runners are kept in the sibling repository
`/home/kellen/wheel_leg_sim2sim`; this repository contains the training tree
model and its parity asset only.
Its current leg adapter is intentionally approximate and must not be treated as
the final parallel-link torque mapping.

Solver settings can be compared explicitly:

```bash
cd /home/kellen/wheel_leg_sim2sim
python sim2sim_closed_policy.py --help
```

The runner prints `closure_max` in metres. A lower timestep and higher solver
iterations reduce numerical closure residual, but do not replace the required
active-link torque mapping.

The current 20-second headless zero-command acceptance reached 20,000 physics
steps with two wheel-ground contacts, zero base-ground contacts, symmetric
support, and `closure_max` about 1.13 mm. Motion-command sim2sim remains an
open follow-up; the present static trim/command gate is explicitly an adapter
baseline, not the final locomotion controller.
