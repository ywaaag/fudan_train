# ±2 m/s 停车保留实验（2026-09-20）

已验收候选：`plane/logs/wheel_legged/Sep19_23-08-55_legacy_speed2_stop_20260919_230848/model_1400.pt`。
完整证据及SHA见 [JSON](data/legacy_speed2_passed_20260920.json)。这是稳态平移通过，
不是最终±4m/s、±4rad/s、转弯/反向切换目标完成。

## 继承和受控变量

从 LEGACY_SPEED2 run `Sep19_22-05-02_legacy_speed2_20260919_220453/model_900.pt`
完整继承actor/encoder/critic/std及两套Adam；源900三种子运动24项通过，zero约+.104m/s失败。
仅将采样改为40% exact zero，其余八个±.5/±1/±1.5/±2端点各7.5%；保留reward、noise、
随机化和optimizer，yaw=0、height=.40。4096env、seed23、追加500iteration。

## 上一轮无效对照及修正

`legacy_speed2_stop_20260919_225419`（v1）不能作为有效的±2停车对照：
zero_retention旧分支硬编码±.5/±1，遗漏±1.5/±2；runner验收列表又被默认五命令覆盖。
此前“保留全部速度端点”的说明不准确。该轮模型和结果保留，但不用于归因或升速。
v2修复采样分支及验收列表；train.py亦修正LEGACY_SPEED2应从700、STOP应从900的状态检查。
400槽、多种offset测试验证zero160次、八端点各30次。相关4项测试通过；80env一iteration
smoke通过，逐项完整resume验证通过，生成901。修正版仍从原900启动，不使用v1模型。

## 实际验收

训练命令（启动新实验，不是查询）：

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_h3_low_speed.py --profile LEGACY_SPEED2_STOP
```

job：`plane/outputs/legacy_speed2_stop_20260919_230848`。训练约298秒；已结束，无需重启。
seed19/37/53，各命令16env、25秒、5秒warmup、level1随机化、deterministic。
model1000为24/27；model1400为27/27。全部无失败/timeout/非轮触地，双轮接触100%。

| 命令m/s | 三seed平均实际vx | 三seed平均MAE |
|---:|---:|---:|
| 0 | −.03924 | .03924 |
| −.5 | −.56383 | .06383 |
| +.5 | +.46950 | .03050 |
| −1 | −1.09652 | .09652 |
| +1 | +.95460 | .04540 |
| −1.5 | −1.58964 | .08964 |
| +1.5 | +1.43892 | .06108 |
| −2 | −2.04516 | .04516 |
| +2 | +1.93674 | .06326 |

−1接近.10门槛，zero仍有慢漂移，不能称为位置保持或高裕量策略。
ONNX位于该job的`model_1400.onnx`；verify_onnx.py --batch=256通过，最大误差5.7220459e-6，
详细输出`onnx_verification.log`。未进行MuJoCo sim2sim，不作跨模拟器成功声明。

下一步保留1400与旧700，先建立低速yaw及切换验收，再训练转向/反向课程；不能直接宣称±4完成。

相关实现：`plane/wheel_legged_gym/envs/base/command_sampling.py`、
`plane/wheel_legged_gym/envs/wheel_legged/legacy_speed2_stop.py`、
`plane/wheel_legged_gym/scripts/train.py`、`tools/run_h3_low_speed.py`；
回归测试：`plane/tests/test_speed2_stop.py`。
