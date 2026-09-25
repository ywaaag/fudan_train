# model9500派生实验：已获用户授权

最终验收已完成：同一派生model_10000在显式斜坡协议下通过五项闭链目标，并通过
前进/后退高速后停车。详见[正式验收记录](sim2sim_10000_acceptance.md)和
`docs/data/sim2sim_goal_10000_20260922.json`。四种直接±4阶跃仍失败，不能推广为
急停或任意命令切换成功。下面保留完整实验过程。

用户已明确纠正goal：允许调整训练配置、训练派生模型缩小sim2sim/sim2real gap。
2026-09-22已成功重新创建active goal，明确同一派生策略停车、前后±4m/s、自转±4rad/s
经独立闭链验证全部通过才完成。不再受旧goal固定9500表述限制，不再等待微调许可。
本轮已实施控制/切换配对实验；不预称已解决闭链问题，原策略、XML、映射及保护保持。

## 源预检

- checkpoint：`plane/logs/wheel_legged/Sep21_14-37-19_motion_goal_20260921_143444_r01_basic_motion/model_9500.pt`
- SHA256：`6f120e632d6497926819ced8d25181c5ad08c7dd882ae4e4ed0f1e89cf713c23`
- ONNX SHA256：`9a0cbf8c01f62840019d34202b0e8069191e4f38c3bfce9341f140183499f788`
- checkpoint实际actor/critic Adam LR=1e−5；encoder Adam LR=.001；不是manifest名义actor LR=.001。
- actor std：[.22887,.35567,.24649,.24808,.36405,.24303]；完整状态和moments存在。
- 当前已有摩擦.6～1.4、惯性.9～1.1、质量−1～2kg、Kp/Kd/电机扭矩.95～1.05、
  默认关节角±.03等随机化。不能把“开启随机化”当作新改动，或声称没有随机化导致失败。

## 建议首个单变量对照

先测试命令在episode内切换（原来hold until reset），每5秒只在原basic_motion的
50槽命令bank中重采样，包含停车/纯直行/纯自转，不引入组合转弯或高度调整。
与同源不切换的控制组对照；各1iteration smoke+500iteration，4096env，同seed23。
这检验动态接触适应假设，不保证修复−4m/s稳态误差。不得同时改变reward、encoder LR、
摩擦、PD或接触参数。完整继承实际optimizer LR，不重置成名义配置。

每组保留全部100iteration checkpoint：先Isaac单seed筛查，候选做三seed全部25命令、
75项稳态与直行几何≤3cm；然后ONNX256输入对照，再完整闭链同协议五项验证。
闭链至少包含zero、±1/±2/±4，固定tree_zero初始状态、相同2+10+2+20s输入时序；
另记录历史±4阶跃和正向30秒爬坡，不能只挑更易通过的协议报告。
重点看负向yaw接触丢失是否消失、后退vx MAE是否≤.10、正向/停车是否回归。
不得在sim2sim端偷偷改命令、加入LQR或放宽15mm腿差保护。

若源端改善但闭链未改善，停止相同重跑；再考虑基于确切动力学差异设计另一独立
随机化消融，不对实际XML改参数来拟合策略。不得将当前映射有限差分通过解释为
完整动力学等价。最终所有结论以独立闭链证据为准。

实施：motion_goal.py增加command_switch_seconds=5（仅basic_motion）和runner CLI；
其余模型reward/optimizer/physics配置不变。新增test_motion_switch_recipe.py证明差异
只涉及hold_command_until_reset/resampling_time，6项相关回归通过。
新增tools/run_sim2sim_adaptation_pair.py顺序执行同源control/switch5，每组smoke+500iteration，
沿用五保存点筛查、最佳两个三seed源端验收。每组独立日志与checkpoint，不重复来源相同实验。
训练组结束后状态training_pair_complete_pending_closed_chain_validation不是goal完成。
当前配对job为`plane/outputs/sim2sim_adaptation_20260922_081907`，control子job为
`plane/outputs/motion_goal_20260922_081909`；现场已确认supervisor及子进程存活。
控制组已完成：最佳三seed候选9600运动70/75，安全/几何通过（最大镜像均值2.03cm），
失败为±4rad/s原地自转；没有accepted，不替换9500，也不将该点用于最终闭链通过证明。
switch5组子job为`plane/outputs/motion_goal_20260922_082857`，已自动从原9500独立开始，
未使用控制组9600。配对训练尚未结束，不能据控制组结果断言切换组有效。
监督器STOP文件转发子runner，保留checkpoint；尚未实现跨进程任意中断自动恢复。

闭链接续：新增tools/validate_adaptation_candidates.py等待该pair完成，核对supervisor
存活，不用状态文件本身推断进程存活。每组只有accepted存在且ONNX存在才运行相同
closed_ramp_v2协议；未过源端门槛标记not_run_source_acceptance_failed。结果写pair目录
closed_validation_status.json，每组原始闭链日志与JSON均保留。
validate_closed_ramp.py新增--policy显式输入、记录并逐结果核对SHA，默认旧9500不变。
完成状态completed_pending_comparison只是实验结束，不自动标记goal完成。
本轮代码py_compile/diff检查通过；control运行manifest现场核实完整9500状态继承、
实际LR=1e−5，已进入500iteration正式训练。

## 配对结果与后续隔离实验

配对实验现已结束，两组均无accepted，闭链接续器分别记录
`not_run_source_acceptance_failed`。切换组9600和10000单seed运动25/25通过，但
直行最大镜像均值分别11.89cm、11.31cm；其余三个保存点也未通过姿态保留。
不能用运动得分掩盖姿态退化，也没有新的闭链通过结论。原9500继续保留。

下一轮单独测试冻结encoder更新：从原9500开始，固定命令不切换，同seed23，
1iteration smoke后500iteration，与上述control对照。只跳过encoder Adam step，
完整加载且保留encoder参数、LR和Adam moments；actor/critic更新、奖励、随机化、
命令bank均不变。此试验用于隔离续训漂移，不预判encoder已被证明是根因。
新增runner选项`--freeze-motion-encoder`，仅允许basic_motion；train.py在完整
resume校验通过后启用已有PPO freeze机制。应实测smoke checkpoint encoder和
其optimizer与源完全相同，actor已更新，之后再判断独立验收。

冻结组job：`plane/outputs/motion_goal_20260922_083813`。现场smoke验证已通过：
model_9501的encoder逐tensor与9500完全相同，encoder Adam状态完全相同，actor
已变化，manifest冻结标记为true。证据为该job的`freeze_smoke_verification.json`。
正式500iteration已启动；`validate_adaptation_candidates.py --single-job`已挂接该job，
只有源端完整通过并导出/核对ONNX后才自动闭链验证。6项相关配置回归通过。

冻结组现已完成：9600/9700/9800/9900/10000各单seed运动25/25，但几何最大均值
依次4.52/5.20/4.84/5.11/4.78cm，全部姿态保留失败；没有accepted，不进入闭链。
这支持继续隔离姿态退化，但不能证明encoder冻结已经解决sim2sim gap。
下一消融仍从原9500，同seed23、冻结encoder、固定命令、500iteration，仅把现有
几何reward权重−.1改成−.2，容差/尺度/yaw条件不变。原9500和所有失败点保留。
CLI新增`--geometry-weight=-.2`，限定冻结encoder的basic_motion实验；来源reward
校验按显式spec检查该项，其他reward检查不放宽。critic/Adam只作新reward初始化。
该消融job为`plane/outputs/motion_goal_20260922_084744`，已通过1iteration smoke，
正式500iteration进程现场存活。smoke manifest确认freeze=true、weight=−.2，
完整resume模型和两组optimizer逐项校验通过；自动闭链接续器已启动。

该轮训练及源端验收现已完成。选中
`Sep22_08-49-16_motion_goal_20260922_084744_r01_basic_motion/model_10000.pt`，
三seed运动75/75、几何最大均值2.740cm通过；原9500继续保留。
ONNX256输入一致性max_abs_error=5.7220459e−6，检查通过。
ONNX SHA256：`1177e2c46d3a831f98e7b4548b0cef4c0b8de8ea9af57ae1c3efddac63ae2468`。
相同闭链协议已开始：`plane/outputs/closed_ramp_v2_20260922_085731`；
源端通过不代表闭链通过，必须继续检查五项目标及完整物理保护结果。

同策略阶跃补测：`plane/outputs/closed_speed_20260922_090051`，直接复位后下达
±4命令，四项都触发轮子contact保护；+4vx/−4vx/+4yaw/−4yaw分别仅完成
41/19/52/44个1ms步，均无稳态样本。不是稳态速度结论，更不是通过。
此次扩展`tools/validate_closed_speed.py --policy ... --target-only`，显式记录和
逐结果核对ONNX SHA；未改变原保护、物理dt或控制器。与斜坡结果分别报告。
斜坡job仍运行中，已完成zero和+1vx：速度MAE分别.01442/.00524m/s，
各34000步、20000稳态样本、物理和tracking均通过，其他结果尚待完成。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_sim2sim_adaptation_pair.py
```
