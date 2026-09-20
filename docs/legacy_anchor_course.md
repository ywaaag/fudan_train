# Legacy分支的固定命令课程（2026-09-19）

当前normalized分支的zero40采样和关闭观测噪声两个尝试都退化后停止。
下一实验改从 `Sep19_11-22-46_legacy_urdf_20260919/model_500.pt` 完整续训，
保持它实际使用的reward、PPO、随机化、观测noise和std，不覆盖任何历史模型。

唯一改变的配置组是command curriculum：原连续vx/yaw混合且自动升速，改为
episode内固定的translation命令，zero20%、±.1各10%、±.5/±1各15%、yaw=0、
height=.40；禁用自动升速。不是把reward、noise或学习率又一起换掉。

Profile `LEGACY_ANCHORS`保留源实际混合legacy recipe中的zero penalties、当前URDF
适配和当前legacy代码差异，不称为原Fudan严格复现。源optimizer的自适应LR保留，
实际起始LR为约0.00058528，不强制重置到配置默认0.001。所有网络tensor/std/两套Adam
逐项验证一致；源SHA保存在 `docs/data/legacy_anchor_source_20260919.json`。

64env×1iteration smoke通过，完整state检查通过。5项legacy/profile/sampler测试通过。
正式计划4096env、seed23，500→1000追加500iteration，验收600/1000各三seed。
验收采用当前通用method_v1固定命令环境（0/±.5/±1、25秒/5秒warmup、16env、level1），
以统一的稳定性/速度门槛筛选；这与原legacy报告的legacy termination不同，不能
将两报告的通过率当成严格相同protocol对照。训练reward不参与确定性动作推理。

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py --profile LEGACY_ANCHORS
```

该命令启动新实验；中断先核对进程、status及checkpoint，不重复启动。
正式低速基线仍为H3迁移model100；legacy500是实验来源，尚未通过停车/后退门槛。
没有command offset、真实速度替换或其他隐藏补偿，完整目标仍未完成。

新增：`envs/wheel_legged/legacy_anchors.py`、`plane/tests/test_legacy_anchors.py`、
源SHA记录和本文。修改：`envs/wheel_legged/policy_experiments.py`、`scripts/train.py`
（envs/scripts均在plane/wheel_legged_gym）、`tools/run_h3_low_speed.py`。

## 已找到通过点：model700

run `Sep19_12-59-33_legacy_anchors_20260919_125925` 完成500→1000。
末尾1000虽12/15但zero失败；补测700/800/900后，对700完成三seed独立验收，
**0/±.5/±1共15/15通过**。选择700，不以latest代替最佳。
证据及SHA：`docs/data/legacy_speed1_passed_20260919.json`。

model700三seed实际vx均值约：zero+.0406、-.5=-.5069、+.5=+.5478、-1=-.9995、+1=1.0163。
全部速度/高度/yaw/接触/失败/饱和门槛通过。仅稳态平移验收，不代表yaw/turn/反向切换完成。
ONNX：`plane/outputs/legacy_anchors_20260919_125925/model_700.onnx`，256输入数值检查通过，
最大绝对误差3.8147e-6。当前平移基线更新为700，旧model100与全部分支保留。

## 下一阶段LEGACY_SPEED2

从已验收700完整续训，保留实际Adam学习率、网络/std、reward、noise/domain randomization。
只扩展命令range至±2，anchors为.5/1/1.5/2，每方向每端点7.5%，zero20%，±.1各10%。
自动升速关闭，yaw0，高度.40。64env一iteration smoke通过；2项配方/采样回归测试通过。

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py --profile LEGACY_SPEED2
```

追加500到1200，验收800/1200各三seed，命令0/±.5/±1/±1.5/±2，共27项必须全过。
新增`envs/wheel_legged/legacy_speed2.py`、`plane/tests/test_legacy_speed2.py`，
修改policy_experiments.py/train.py及run_h3_low_speed.py；机器人资产和控制接口不变。
