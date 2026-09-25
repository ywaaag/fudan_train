# 同策略五项闭链验收（2026-09-22）

在明确的斜坡输入协议下，派生model_10000通过停车、直线±4m/s、原地±4rad/s，
以及前后高速运动后停车。不是任意阶跃、急停、快速切换或sim2real验收。
直接从复位下达四种±4阶跃仍失败，详见下方范围限制。

## 模型和来源

- checkpoint：`plane/logs/wheel_legged/Sep22_08-49-16_motion_goal_20260922_084744_r01_basic_motion/model_10000.pt`
- checkpoint SHA256：`f645394a0d49ce42a5908aee3394caa5c3836d9f618750e521e931b51d57fd39`
- ONNX：`plane/outputs/motion_goal_20260922_084744/accepted_basic_motion.onnx`
- ONNX SHA256：`1177e2c46d3a831f98e7b4548b0cef4c0b8de8ea9af57ae1c3efddac63ae2468`
- 从保留的原9500完整warm start，seed23、4096env、500iteration；冻结encoder更新，
  几何惩罚−.2，固定原basic_motion命令bank。奖励改变，critic/Adam仅作初始化。
- 源端三seed×25命令75/75，几何最大均值2.740cm；ONNX256输入一致性最大误差5.722e−6。
- 原9500及所有控制/失败分支、checkpoint、manifest、原始JSON均保留。

## 协议及结果

保持25D observation /125D history /6D action；根高0.40m tree_zero初始化。
独立完整闭链MuJoCo：固定1ms物理步长、100Hz policy；原闭链映射、PD、限矩和
每步接触/闭链/腿长差保护保留。没有运行时LQR、静态配平、命令偏移或隐藏门控。
初始化几何求解不参与运行时控制。

输入为2秒零命令、10秒线性升速、2秒稳定、20秒测量，总34000步；各项20000稳态样本。
数值均为实际仿真测量，不用reward推断通过。

| 项目 | 平均vx (m/s) | 平均yaw (rad/s) | vx MAE | yaw MAE | 结果 |
|---|---:|---:|---:|---:|---|
| 零命令停车 | −.01442 | .00606 | .01442 | .00606 | 通过 |
| 前进4 | 4.07984 | −.00477 | .07984 | .00477 | 通过 |
| 后退−4 | −4.09667 | .01501 | .09667 | .01501 | 通过 |
| 正向自转4 | −.03487 | 4.01568 | .03487 | .01659 | 通过 |
| 反向自转−4 | .03764 | −4.01851 | .03764 | .01851 | 通过 |

原门槛：移动vx MAE≤.10，零vx目标≤.05，yaw MAE≤.10，高度MAE≤.03。
五项均完整完成，轮子持续有接触、无非轮触地，最大闭链残差约1.423mm。
后退误差距离.10门槛仅约.0033m/s，不应宣称具有大幅鲁棒性余量。

另外分别从±4m/s保持到第34秒，再显式10秒线性减速、2秒稳定、20秒停车观察，
总66000步。两项均通过；停车vx MAE分别.014421/.014419，yaw MAE约.00607。
轨迹确认减速前确实达到目标速度。原始保护逐1ms检查；100Hz记录轨迹最大腿长差
约1.75mm，该采样最大值不能替代逐步保护检查。

## 证据与复现

- 机器可读审计：`docs/data/sim2sim_goal_10000_20260922.json`，包括逐项条件、原始证据SHA。
- 停车：`plane/outputs/closed_ramp_v2_20260922_085731/zero.json`。
- 四项高速及两项运动后停车：`plane/outputs/closed_target_trace_20260922_10000/`。
- 同协议完整速度梯度验证：`plane/outputs/closed_ramp_v2_20260922_085731/`；独立后台补充验证，
  不把其尚未完成的条目计为通过。本报告高速结论来自已完整完成的目标轨迹测试。
- MJCF SHA256：`663e121ef9aed4bfab09ff0d3d98321b7ce77d80b63584235e4da691bd3aebf3`。
- runner SHA256：`04af1b89a6f3690d7002182c53438efd1b7d5add7d1c29bd3bbff51e283e1cdd`，
  与原9500失败基线一致；XML、闭链映射和安全模块未改。

仓库根目录生成审计：

```bash
/home/kellen/anaconda3/envs/robot/bin/python tools/summarize_sim2sim_goal.py --candidate 10000
```

独立复现后退（输出文件必须不存在；GUI可追加`--viewer`）：

```bash
/home/kellen/anaconda3/envs/robot/bin/python tools/probe_closed_initialization.py \
  --policy plane/outputs/motion_goal_20260922_084744/accepted_basic_motion.onnx \
  --out /tmp/model10000_reverse4_review.json --initialization tree_zero \
  --steps 34000 --settle-seconds 2 --ramp-seconds 10 \
  --metrics-warmup-seconds 14 --forward -4 --trace
```

其余目标改为`--forward 4`、`--yaw 4`、`--yaw -4`或不传运动命令。
运动后停车在±4直行测试上追加`--return-to-zero-after 34 --return-ramp-seconds 10`，
将steps改66000、metrics-warmup-seconds改46。

## 适用范围

直接±4阶跃四项仍在41/19/52/44ms触发轮子接触丢失，证据在
`plane/outputs/closed_speed_20260922_090051/`。因此必须显式使用上述输入协议，
不能声称0.3～0.5秒响应、1～1.5m急停、任意切换、位置锁定、多高度或实机通过。
本轮goal允许记录斜坡协议，其五项运动能力已获证据；下一阶段可单独训练动态切换。
