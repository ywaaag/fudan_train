# 机身高度命令课程（2026-09-20）

## 用户决定暂停多高度，恢复转弯主线

双高度job `dual_height_20260920_132700`完成500iteration，但仅3/6通过：
35cm命令实际44.84cm、45cm命令实际44.54cm；无failure、停车稳定。
单一35/45cm分别可学会，不代表同策略条件跟踪已实现；根因尚未隔离。
未进入升降切换，未保留或验收运动能力，因此不使用高度实验模型作为转弯来源。

用户明确要求提交当前Git状态并暂不训练多高度。全部高度流程保持停止，代码、日志、
checkpoint和诊断结论保留。转弯从已验收的motion_goal r24 model6900继续，height固定.40。

## 固定高度通过，推进单策略双高度

隔离job `fixed_height_diagnosis_20260920_124510`已完成：fixed35/fixed45末点7400
均通过三种子全部3项固定高度停车门槛。实际平均高度约35.23/44.95cm，vx MAE约
.0082/.0112m/s。各组无failure、双轮接触100%，ONNX数值一致性通过。
两组的7100/7200/7300也分别通过三种子；7000未通过首seed。
来源相同6900，两组独立训练，不是策略拼接。

这证明当前tree模型、控制接口和原高度reward能学习单个35/45cm稳定停车；
不能推广为闭链实机可行、运动能力保留或全部高度可达。先前联合训练失败更可能涉及
条件学习与多任务优化，但此实验尚未区分二者，不能认定唯一根因。

下一实验`dual`：从原6900重新完整续训，同一网络采样(0,0,.35)/(0,0,.45)各50%，
保持其余原配方、reset=.40、训练seed23。先1iteration smoke，再500iteration；
三种子×两命令共6项稳态验收。若通过自动导出ONNX，并额外评估不reset的
35→45、45→35升降：第5秒切换，第10秒起统计稳态，逐环境10Hz响应trace；
高度MAE≤15mm、停车/姿态/接触门槛保留，同时要求vx/yaw/height均在3秒内进入并
持续保持相应误差带。若稳态不通过暂停，不宣称固定高度专家的成功等于条件策略成功。
升降通过后才重新引入低速运动，原6900保持为运动基线。

入口`tools/run_dual_height.py`；状态写独立dual_height job的status.json。
本次job：`plane/outputs/dual_height_20260920_132700`，已验证smoke完成并进入dual_train，
完整网络及两套Adam检查通过，初始iteration6900，实际bank恰为两个高度各50%。
新增run_dual_height.py；修改height_course.py、run_height_course.py、test_height_course.py。
9项测试通过，包括双高度50/50覆盖、现有高度课程和响应统计。py_compile/diff检查通过。

## 当前：固定高度能力隔离（用户已授权连续推进）

5秒高度切换三轮末点75/60/87项通过，总135；均未保持全部旧运动门槛，自动暂停。
因此停止同类采样/随机种子重试，先区分固定高度控制与条件学习/任务干扰。

新job：`plane/outputs/fixed_height_diagnosis_20260920_124510`，
实时`status.json`、`progress.md`，STOP文件可停止子流程并保留所有模型。
`tools/run_fixed_height_diagnosis.py`按顺序执行fixed35和fixed45，两者都从原
`Sep20_03-42-29_motion_goal_20260920_002043_r24_yaw4/model_6900.pt`独立完整续训，
绝不用35cm实验checkpoint初始化45cm实验。两组训练seed23，80env smoke后4096env、
500iteration。只改变命令为(0,0,.35)/(0,0,.45)，原height reward gain1、噪声、
Adam/std、控制接口和reset .40m保持不变，无IK/LQR或动作补偿。

两组即使其中一组验收失败，也继续完成另一组。每组末点使用seed19/37/53、16env、
25秒/5秒warmup，要求高度MAE≤15mm、vx MAE≤.05、yaw MAE≤.1、无failure/timeout/
非轮触地、双轮接触≥.99、饱和≤.01。随后补筛7000/7100/7200/7300：先seed19，
通过才追加37/53。不同验收种子不是多次独立训练，不能据此证明训练可重复性。

评估器额外记录关节平均/最小/最大位置和施加力矩绝对最大值。力矩为policy步快照，
不能取代全部physics子步preclip saturation计数；轮关节位置为累计旋转角。
如果两高度都通过，状态为fixed_heights_passed_ready_for_conditional_design，
下一步才设计同策略双高度/升降，再逐级恢复运动。若未通过，记录失败与姿态证据，
不能据500iteration失败直接断言机械不可达。诊断模型不替换运动基线，不宣称运动保留。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_fixed_height_diagnosis.py
```

新增run_fixed_height_diagnosis.py；修改height_course.py、run_height_course.py、
evaluate_policy_comparison.py、test_height_course.py。7项测试、py_compile和diff检查通过。
已现场确认fixed35 smoke完成并进入正式训练。旧6900及全部失败实验保留。

## 中间点补筛与高度切换对照

micro三轮（seed23/37/53）末点72/60/72项通过，总135项；均未保留全部旧运动门槛，
保留原6900。第三轮39/40/41cm停车实际约40.34/40.28/40.19cm，仍近常量且方向略反。
新增`tools/screen_height_checkpoints.py`逐一补筛三轮7000/7100/7200/7300共12个点，
每点先seed19完整45命令，不通过则停止该点；通过才追加37/53。不能将seed19筛查称为
三seed完整验收。job：`plane/outputs/height_screen_20260920_115833`。

连续流程job：`plane/outputs/height_continuation_20260920_120014`。
它先等待补筛；若发现通过点停止并报告候选，避免覆盖好模型。若无通过点，启动高度
时序对照：仍原6900、micro、repeats4、gain1，唯一课程变化是非40cm样本每5秒在
39/41cm间切换。vx/yaw和40cm保留样本不变，不reset、不注入动作或几何补偿。
目的：检验episode内命令变化能否促使条件响应，不能预先认定因果或宣布修复。

每次先跑真实15秒无更新rollout，逐步断言命令进入observation，统计不reset的高度
切换，断言切换不改vx/yaw也不影响40cm保留样本；没有观测到切换即报错停止。
再1iteration smoke、500iteration、135项固定高度验收；最多3轮重试，无改善暂停。
通过也只代表固定命令micro通过，不代表动态升降验收或30～45cm目标完成。
已修复子job run_name在不同supervisor间重复的问题：加入parent job名称保证新run可区分。

新增：`tools/screen_height_checkpoints.py`、
`plane/wheel_legged_gym/envs/base/height_commands.py`、`plane/tests/test_height_commands.py`。
修改：height_course.py、legged_robot.py、run_height_course.py、continue_height_course.py、
audit_height_rollout.py。本次7项相关测试通过；script py_compile与git diff --check通过。
首次screen脚本拼接行SyntaxError已修复后才启动，不涉及训练权重变更。
补筛最终完成12/12，各点均未通过seed19的完整45命令检查，未追加seed37/53。
新连续流程首轮真实审计观察到960次未reset高度切换、45种命令组合；smoke完成，
已进入`round01_micro/micro_train`正式训练。数据在该子job的micro_schedule_audit.json。

## 第三轮失败与连续小范围课程

reward×4的job `height_course_20260920_100352`结束后22/135通过，476次failure重置；
其中463次在(4m/s,0rad/s,.4m)，其余在原地+3/+4rad/s。不能将其解释为高度端点
本身必然不稳定；本轮全局奖励加强严重损伤旧高速能力，路线停止，不使用其7400续训。

用户要求连续推进。新增`tools/continue_height_course.py`，从原6900恢复gain=1、
repeats=4，先改为micro=39/40/41cm。相对第二轮只改变高度范围，不混入第三轮gain4。
micro高度MAE≤5mm（其余阶段15mm），因此常量40cm不能被当成39/41跟踪成功。
保留40cm完整31个运动命令。每轮1iteration smoke、500iteration、三seed135项验收。
每轮来源、seed、checkpoint、结果与回退决定写入supervisor status.json/progress.md。

连续顺序micro→near→middle→full。通过才升范围；未全过的候选只有无任何failure/
触地/接触/饱和等安全问题、40cm全部旧运动门槛通过，且分数改善，才允许同阶段继续。
其余回退保留source；重试仅改变训练seed为23/37/53，不能把随机重试当确定修复。
本次supervisor：`plane/outputs/height_continuation_20260920_110521`，首轮子job为
`round01_micro`，smoke完成后已进入正式训练；查询该目录status.json/progress.md。
连续3轮失败或无改善暂停诊断，最多12轮。不直接进入动态升降或宣称全高度高速运动成功。
该初版只验收每轮末点，可能漏掉较优中间checkpoint，暂停后应补筛，不能宣称整条分支均失败。

停止方式：在supervisor job目录创建`STOP`，会转交子runner并停止其训练/验收子进程。
使用height_continuation.lock和已有训练锁防重复启动；此脚本无自动崩溃恢复，重启前须查进程。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/continue_height_course.py
```

新增continue_height_course.py、test_height_continuation.py；修改height_course.py、
run_height_course.py、test_height_course.py。6项相关测试通过，验证micro端点、旧能力
样本保留、常量高度不能通过、失败候选拒绝。未覆盖或删除原模型和三轮失败证据。

## 第二轮结果与高度奖励对照

加权采样job `height_course_20260920_092618`完成500iteration，30/135通过，自动暂停。
37.5/40/42.5cm三组停车高度均约43.54cm，vx分别+.075/+.081/+.090m/s；
全验收2次failure。仅提高可变高度采样没有解决条件跟踪，禁止升范围。

动作审计：`height_sensitivity_audit.json`（上述job内），以真实rollout状态为固定背景，
只改当前及五帧height通道37.5→42.5cm，观察actor mean（不执行反事实动作）。
各通道平均绝对动作差约.023～.048，明显非零；左髋平均变化约−.047，右髋约−.003，
未形成明显一致升降模式。sampled rollout低高度实际约43.44cm，height reward约.04。
这说明输入有作用但未学成目标跟踪，不证明noise或网络容量是根因。

新增`tools/audit_height_update.py`执行同源6900的真实15秒以内rollout及一次PPO更新，
不保存checkpoint。证据`source_ppo_audit.json`：三个height组有7680/9216/7680样本，
标准化advantage均值约−.559/+.921/−.546，说明当前40cm行为在该批数据上更有优势。
更新前后height动作差异非零且发生变化，排除“完全没有更新”；单批advantage混合其他
reward/命令差异，不能用它断言critic错误或全部失败原因。

第三轮从同一6900出发，保持variable_repeats=4、原noise/std/Adam/LR，仅改变高度
reward贡献：base_height权重1→4，并将base_height加入unclipped_reward_names。
理由：legacy单项clip=1，若只增权会在近目标区域饱和、抹平误差差别；其他项裁剪不变。
原高度reward函数exp(-error²/.001)不变。奖励改变意味着旧critic/Adam仅作为初始化，
不称为相同任务精确续训。只对base_height放宽源配方检查，其他reward mismatch仍拒绝。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_height_course.py \
 --variable-repeats=4 --stage-limit=1 --height-reward-gain=4
```

本次4项高度测试通过，包括配置差异仅限高度贡献、采样及optimizer保持不变。
第三轮job：`plane/outputs/height_course_20260920_100352`，smoke已完成，正式训练已启动；
manifest核实base_height=4、豁免仅base_height、来源6900网络及两套optimizer状态一致。
仍执行1iteration smoke、500iteration和135项独立验收，未通过不覆盖6900。
修改height_course.py、run_height_course.py、test_height_course.py；
audit_height_rollout.py增加反事实动作探针并正确识别单项裁剪豁免。

## 第一轮失败诊断与受控对照

第一轮near在7400结束，27/135通过，自动暂停；三个高度的停车实际高度均约41cm，
40cm停车vx约−.256m/s。未覆盖原6900，未升阶段。
补测7000/7100/7200/7300（seed19，16env/命令）：37.5/40/42.5cm命令下实际高度分别为
40.42/40.49/40.54、42.05/42.13/42.18、44.34/44.41/44.48、42.50/42.58/42.67cm。
机身可以升高，但输出几乎不区分高度命令；不是只有最后checkpoint退化。
补测是7命令筛查，不替代完整验收。

`tools/audit_height_rollout.py`在真实训练profile下以sampled action/noisy observation
无更新运行15秒，warmup5秒；逐步断言命令进入obs，覆盖全部45个唯一联合命令。
37.5/40/42.5cm样本分别40000/192000/40000（env×step），无重置。
高度平均40.65/40.23/40.73cm，高度reward贡献每秒约.377/.966/.727；
nominal_state约−.006、orientation约−.018～−.027。不能据此声称高度奖励没执行、
采样遗漏或nominal强锁40cm，也不能把单次诊断推断为已证明的因果机制。
证据：原job下`training_rollout_audit_reset.json`、`screen_7000.json`至`screen_7300.json`。
初版诊断忘记env.reset，产生`training_rollout_audit.json`（零命令、零有效高度样本），
该文件无效，保留但不作证据；正式trainer已有reset，没有因此改变训练。

下一轮仍从原6900完整续训，只改变可变高度命令重复次数1→4：20/68≈29.4%变为
80/128=62.5%，40cm原48个slot及全部31个唯一运动命令保留。
奖励、noise、学习率、std/Adam来源与首轮相同。该变化检验高度条件训练暴露不足的假设，
不把假设当根因。只near阶段，smoke后500iteration，失败暂停，禁止自动升范围。

对照job：`plane/outputs/height_course_20260920_092618`，已验证smoke完成并进入正式训练，
manifest中的reward_scales、optimizer、实际起始LR及resume_verification与首轮完全一致。

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_height_course.py \
 --variable-repeats=4 --stage-limit=1
```

新增audit_height_rollout.py，修改height_course.py/run_height_course.py并扩充
test_height_course.py；采样加权测试验证命令集合不变、40cm次数不变、其他次数精确4倍。

用户新增目标：同一策略能按高度命令在30～45cm调节机身，并保留运动能力。
先于高速后退转弯诊断，建立独立分支；原model6900不覆盖。

## 已核对的接口与基线

高度是base_link/root原点相对地面的高度，不是机壳底部离地间隙。
当前25D observation的第8通道（零起始）已是height command，包含于5帧history；
legacy `_reward_base_height`按实际高度与命令之差计算exp(-error²/.001)。
因此无需改变25D/125D/6D接口、URDF或控制器。该奖励在较大高度误差处很弱，
首轮从40cm附近开始，而非把10cm误差直接交给同一局部奖励学习。

URDF静态平面运动学：令wheel中心x=0、z=.06−height，绕各腿Y轴求两关节角。
左腿30/35/40/45cm的解（rad）约为(.349,−.571)、(.179,−.301)、(0,0)、(−.207,.363)，
右腿接近相同；均在URDF±3.14内，末端位置残差<1e-8m。
这仅是tree模型几何可达检查，未证明碰撞间隙、闭链实机可行性或动态稳定性。
训练不注入这些角度，不添加IK/LQR补偿，reset仍为原40cm姿态。

基线：`plane/outputs/height_baseline_20260920/model6900_seed19.json`，
16env/高度、seed19、25秒/5秒warmup、level1随机化、deterministic。
命令30/32.5/35/37.5/40/42.5/45cm，实际平均高度分别约
40.99/40.94/40.74/40.52/40.03/39.72/39.75cm，无失败重置。
说明现有模型没有高度跟踪能力，不是修改配置range便已成功。

## 训练及门槛

来源：`Sep20_03-42-29_motion_goal_20260920_002043_r24_yaw4/model_6900.pt`。
profile `HEIGHT_COURSE`只改变(vx,yaw,height)联合命令采样；所有reward、PPO、noise、
物理、std及两套Adam保持来源语义。每次完整resume校验SHA、网络tensor和optimizer。

阶段near=37.5～42.5cm，middle=35～45cm，full=30～45cm，间隔2.5cm端点。
每阶段保留40cm时全部已通过motion_goal/yaw4命令，其他高度先采样停车、±.5/±1m/s、
原地±.5rad/s。尚未承诺所有高度都能±4m/s/±4rad/s，更不能直接宣称连续高度或升降切换已通过。
每阶段80env×1iteration smoke后4096env、seed23短训500iteration；末尾三seed19/37/53验收。
高度MAE≤1.5cm（比原3cm更严格），速度/角速度/接触/失败门槛保持。全部通过才扩展范围；
失败暂停诊断，不继续用失败模型升阶。通过的阶段导出ONNX并检查256输入一致性。

实际job：`plane/outputs/height_course_20260920_084138`，状态`status.json`，
每阶段`*_acceptance.json`，本轮启动时先通过smoke后已进入near_train。

```bash
# 查询，不启动新任务
cat plane/outputs/height_course_20260920_084138/status.json
# 启动新实验（已有任务时不要重复执行）
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/kellen/fudan_train/plane \
 LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
 /home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_height_course.py
```

评估器新增`--height-commands`，与vx/yaw按行配对；height_mae改为相对真实目标，
默认仍为.4。`--initial-height-commands`支持不reset的高度切换，trace记录实际height；
切换摘要增加height误差带统计。固定高度通过后还需升降切换、速度组合范围、未知seed、
中间未训练高度和独立sim2sim验收。

新增：`plane/wheel_legged_gym/envs/wheel_legged/height_course.py`、
`plane/tests/test_height_course.py`、`tools/run_height_course.py`、本文。
修改：`plane/wheel_legged_gym/envs/base/command_sampling.py`、`legged_robot.py`，
`plane/wheel_legged_gym/envs/wheel_legged/policy_experiments.py`，
`plane/wheel_legged_gym/scripts/train.py`、`evaluate_policy_comparison.py`，
`tools/summarize_transitions.py`。高度端点、完整运动保留、联合采样及越界拒绝有回归测试。
