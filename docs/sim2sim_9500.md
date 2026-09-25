# model9500 sim2sim历史复核与首次验证

## 五项目标证据审计与后续边界

tools/summarize_sim2sim_goal.py重新读取原始结果，核对policy SHA、输入协议、
完整步数、20000稳态样本、观测维度、接触和速度门槛。
输出docs/data/sim2sim_goal_9500_20260922.json：停车、前进4、正向自转4为3项通过，
后退4与负向自转4未过，goal_complete=false。通过只适用于各自记录的慢爬坡协议。
多轮失败记录全部保留，不把单种子/固定初始化结果泛化为动态与实机成功。

在固定9500权重、固定物理及保护条件下，当前尚无已验证的修复路径；不是证明任何
合法输入轨迹都不可能成功，但继续随意扫爬坡时间不能替代方法上的进展。
映射局部自洽、输入时序及统计问题已修复，尚无证据支持把XML阻尼或接触参数当bug修改。
若要转向训练，应作为model9500派生适应分支，而不是仍称原9500已实现；需用户确认
是否将“当前model9500五项通过”扩展为“9500及其派生模型完成五项”。
可审核的后续方案是保留原9500，以已有Isaac固定物理训练线做小步动态/随机化对照，
每500iteration回归源端稳态及几何，再跑相同闭链协议；禁止增加部署补偿或放宽门槛。

## 实际轨迹上的闭链映射审计（2026-09-22）

新增tools/audit_closed_mapping.py；probe_closed_initialization.py的只读trace增加qpos/qvel。
以相同协议重跑−4m/s记录reverse4_state_audit.json，离线每秒取姿态、两侧共68个样本。
使用约束消元矩阵构造随机主动关节速度及被动切向速度，有限差分轮心位移与有效Jacobian
比较；随机虚拟力矩检查虚拟功率和主动功率一致性。没有改实际运行控制量。
结果reverse4_mapping_audit.json：最大轮心速度有限差分误差1.07e−8，最大虚功差
2.49e−14。支持局部运动学映射自洽，不能推广为动力学完全等价或完整策略稳定。
未发现轮齿比15.7647换算错误；leg gear为1，原主动扭矩裁剪保持。

模型参数对比：tree总质量19.221136kg，full19.221140kg；tree wheel damping/armature
均0，full均.01。该差异会改变动力学，不等于已证实的配置bug。未更改XML、没有为了
压低速度误差修改物理参数。后退超速、负向高速自转接触失败仍未解决，goal保持未完成。

## 匹配输入tree对照与累计轮角限位（2026-09-22）

新增tools/probe_tree_ramp.py，保持原tree物理与控制器，仅提供相同2秒站稳、10秒爬坡、
14秒起测量输入。输出closed_yaw_boundary_20260922/tree_ramp_reverse4_angles.json。
约22秒右轮超过+1000rad时首次失接触，随后左轮碰限位、模型倒地：21.9秒右轮993.96rad
双轮接触，22秒右轮1000.46rad、右轮接触0。编译资产两轮limited=1，range=[−1000,1000]。
因此限位之后的运动不能用于策略稳态对比；本次没有修改轮限位或资产。

14～21秒100Hz采样中tree vx=−3.98889m/s，MAE .01111；闭链保持约−4.12082m/s。
该结果支持进一步检查闭链动力学/虚拟观测，不是最终20秒tree通过，也不能仅凭差异
判定映射有bug。interface_passed只表示接口执行，不能当作物理通过。
左右闭链轮collision size/quat/friction/solref/solimp相同，轴同向，质量差约1e−7kg，
未发现明显左右轮配置错误。新增脚本已实跑并保留轨迹，原sim2sim控制器未修改。

## 30秒爬坡与平滑时序对照

smoothstep对照已完成：−4rad/s和−4m/s均使用2s hold、30s smoothstep、2s等待、20s测量。
−4m/s完整54000步物理通过，但稳态均值−4.12082、MAE .12082，确认误差不是线性
爬坡瞬态造成。−4rad/s smoothstep在29397步后失左轮接触；31.387s是线性爬坡的
失败时刻，不能混用。此前+4rad/s
同样可通过慢爬坡。五项目标仍未完成，未放宽接触或跟踪门槛。

closed_yaw_boundary_20260922/slow30_yaw_4.json：54000步跑满，2秒站稳、30秒线性爬坡、
2秒等待、20秒稳态；20000样本，yaw MAE .01898、vx MAE .03738、height MAE .00461，
物理及跟踪通过。负向同协议在31.387秒、命令−3.918rad/s、实际−3.933rad/s时失左轮接触。
这不是±4阶跃通过，正向成功目前仅证明该慢爬坡协议的能力。

核对XML wheel radius=.06、wheel/ground margin均.002；失接触轮心世界z约.064001m，
接近几何接触边界。roll很小且腿长对称，但仍是接触保护失败。未改margin或接触规则，
不能将其未经证明归类为“数值假错误”。

新增显式--ramp-shape smoothstep，使用u²(3−2u)给定命令，首尾变化率为零，
与线性爬坡一样保持目标、不读取真实速度、不加偏移。用于负向4rad/s及后退4m/s对照。
tools/command_ramp.py及test_command_ramp.py验证单调、边界、目标保持、无过冲。
本次只改输入协议及诊断脚本；全部历史JSON保留，不能选取成功协议后隐去失败协议。

## 2026-09-22 复核：渐进保持结果与边界定位

closed_ramp_v2_20260921_202701已完整完成。每项均为站稳2s、爬坡10s、等待2s、
测量20s；通过项精确20000稳态样本。zero、±1/±2m/s、±1/±2rad/s均物理及跟踪通过。
+4m/s通过，vx=4.06030、MAE=.06030；−4m/s物理通过但vx=−4.12082、MAE=.12082，
超过.10门槛。±4rad/s分别在12.394/11.934s单轮失接触，稳态样本0，不判跟踪通过。
原保护全部保留；五项目标尚未完成。

新增只读--trace记录100Hz命令、速度、roll/pitch、双轮法向力、轮心位置、腿长、闭链
残差及ctrl；首次失接触也强制记录。仅调用原mj_step一次，未改变状态或控制。
输出目录closed_yaw_boundary_20260922；相同协议±3rad/s分别跑满34s，yaw MAE约
.02329/.00999。±4正在按同协议补充失接触轨迹，以区分承载变化与姿态发散。
±4轨迹现已完成：失接触时roll分别−1.75e−5/−6.66e−5rad，左右腿差仅数微米；
左轮法向力在相邻记录由约90N降至0，而右轮约98N。不能称为侧翻，但原接触保护
仍判失败。下一诊断slow30_yaw_±4使用30秒爬坡、34秒后测量20秒，目标仍±4，
用于检查接触丢失是否对输入动态敏感；不得当作修改接触参数或删除检查的依据。

补做MuJoCo parity tree直接阶跃−4m/s，20s终态高度约.0798m，底盘4接触、轮接触0，
已经倒地；日志tree_reverse4.log。interface_passed=true只表示接口，不是物理通过。
该阶跃与闭链渐进协议不同，不能据此判定闭链适配比tree更好或更差。

## 新goal与斜坡统计纠错（2026-09-21）

用户建立新goal：闭链停车、直线±4m/s、原地±4rad/s；保护不变，不用补偿伪造成功。
初版ramp_/hold_测试只可作诊断，不作最终稳态验收：
1. probe额外MAE仍从2秒起统计，即使runner配置7秒，导致混入爬坡。此前−1m/s
   MAE .107“略超门槛”的结论撤回，必须使用修正后的相同稳态窗口。
2. runner reset history原先仍填最终命令、第一步才切成斜坡，输入不自洽。
   已改为以command_schedule(0)构造初始history，未改变25/125维契约。
3. 10秒爬坡只跑10秒没有目标保持段，平均.62不表示目标跟踪不足。

独立sim2sim runner新增可选command_schedule，仅在初始化和策略观测中提供明确的
用户请求时序，不改动作、物理或任何保护。提供有限3D命令校验。默认无schedule行为不变。
修改独立仓库后运行其tests（2 passed）及全部顶层py_compile通过。

新增tools/validate_closed_ramp.py，协议v2：站稳2秒、爬坡10秒、再等待2秒，然后20秒
目标保持统计，总34000 physics步。三项MAE从同一14秒边界计算，完整运行必须精确
20000稳态样本；顺序为zero、各方向±1、±2、±4。物理失败只停止对应方向的升速。
结果保存独立closed_ramp_v2 job，旧日志不覆盖。尚未完成前不宣称五项通过。
本次job：`plane/outputs/closed_ramp_v2_20260921_202701`，status.json记录实时PID及各项结果。

## 逐级直行/自转验证完成

job：`plane/outputs/closed_speed_9500_20260921_160155`。沿用tree_zero初始化，目标
命令从起始时刻直接给出，每项请求20000步/20s，warmup2s。按每个方向逐级1→2→4，
发生物理失败即不继续该方向升速。没有修改独立仓库、XML、控制器或保护门槛。

| 命令 | 物理结果 | vx MAE | yaw MAE | 说明 |
|---|---|---:|---:|---|
| +1m/s | 128步终止 | 未进入warmup | 未进入warmup | 左右腿长差约15.1mm超过15mm门槛 |
| −1m/s | 20s通过 | .0701 | .0265 | 跟踪通过 |
| −2m/s | 23步终止 | 未测 | 未测 | 单轮失接触 |
| +1rad/s | 20s通过 | .0207 | .0032 | 跟踪通过 |
| +2rad/s | 20s通过 | .0238 | .0056 | 跟踪通过 |
| +4rad/s | 51步终止 | 未测 | 未测 | 单轮失接触 |
| −1rad/s | 20s通过 | .0064 | .0165 | 跟踪通过 |
| −2rad/s | 20s通过 | .0069 | .0175 | 跟踪通过 |
| −4rad/s | 68步终止 | 未测 | 未测 | 单轮失接触 |

未测试+2/+4/−4m/s，因为更低级物理检查已失败。runner早停时均值字段0不代表实际
速度为0。腿长错误文字只展示单腿范围，但源码还检查左右差，已核实阈值.015m；
不是单腿超过.18～.40m范围，也不能直接称为跌倒。

新增validate_closed_speed.py，扩充probe_closed_initialization.py只读测量钩子，
记录2秒后每physics步绝对vx/yaw/height误差、全程roll/pitch峰值和control最大值。
实际step仅调用原mj_step一次，结束后恢复hook，不改变策略输入/物理状态/控制量。
跟踪门槛vx .10（纯yaw时.05）、yaw .10、height .03；physical pass必须跑满20s。
读取成功结果核对MAE≥均值偏差，py_compile和diff检查通过。

结论：低速闭链迁移成立，但直接高速起步未通过。下一步对失败方向做显式命令斜坡
诊断，目标和斜坡时间完整记录，区分起步动态失败与高速稳态不可行；斜坡是新的输入
协议，不能冒充原阶跃测试通过。不放宽接触门槛，不注入腿部控制补偿。

## 初始化对照及低速闭链结果

新增训练仓库工具`tools/probe_closed_initialization.py`，独立sim2sim仓库没有改动。
旧闭链初始root=.29975095m，虚拟关节左右均约(.35,−.57248)，与当前训练初始
root=.40m、四腿零关节不同。
诊断tree_zero模式仅在初始化时：机身canonical坐标保持yaw且roll/pitch=0、root=.40；
联合拟合每侧主动/被动6关节，使闭链闭合且轮心等于训练tree的零关节轮心，速度/control=0。
拟合以原装配分支为初值、限制±1.5rad搜索，残差约9.4e−15m；映射虚拟q全零。
不改变XML、不改变策略观测或history契约，不在运行中执行IK/LQR/trim；原full策略步进、
映射和接触检查保持。使用局部诊断wrapper替换初始化，不能称已修改正式部署入口。

| 测试 | 完成 | 平均vx m/s | 平均yaw rad/s | 最大closure |
|---|---:|---:|---:|---:|
| 零命令 | 20s | −.01239 | +.00585 | 1.455mm |
| +.5m/s | 10s | +.49462 | +.01056 | 1.415mm |
| −.5m/s | 10s | −.54709 | +.01431 | 1.356mm |
| +.5rad/s | 10s | −.01417 | +.54078 | 1.493mm |
| −.5rad/s | 10s | −.00831 | −.52958 | 1.549mm |

五项runner physical passed均true，没有放宽接触/闭链门槛。以上为warmup2秒之后的
均值，不是逐步MAE或三seed运动验收，不宣传完整±4sim2sim通过。
新初始姿态使9500从.111秒单轮失接触变为20秒物理通过，支持初始化不匹配为本次
失接触的重要原因；不能据此排除高速下其他映射或物理差异。

日志/JSON均在`plane/outputs/sim2sim_9500_20260921`，zero为
`tree_zero_initialization.*`，低速为forward05/reverse05/yaw_pos05/yaw_neg05。

```bash
/home/kellen/anaconda3/envs/robot/bin/python tools/probe_closed_initialization.py \
 --policy plane/outputs/motion_goal_20260921_143444/accepted_basic_motion.onnx \
 --initialization tree_zero --steps 20000 \
 --out plane/outputs/sim2sim_9500_20260921/new_zero_probe.json
```

--forward/--yaw默认为0，可显式指定；输出若已存在拒绝覆盖。已运行py_compile、diff检查，
核实初始化q为零及完整20000步结果。下一步把初始化方案整理为独立验证器的显式选项，
配套初始化/闭链/虚功测试，再做±1→±2→±4逐级与接触/力矩/跟踪误差验收。

## 用户指定目录的补充核对

`/home/kellen/wheel_leg_mjrl-lqr/fudan_train/README.md`的20秒对称支撑记录针对
早期`../sim2sim_policy.py`，原文明确带static trim/command gate、运动验证未完成。
不能与后来的sim2sim_closed_policy.py --mode full混淆。
`/home/kellen/wheel_leg_sim2sim/README.md`指向独立完整映射runner；执行参数以源码为准。

当前训练parity XML与旧训练目录完全同SHA；当前tree manifest和独立适配器manifest
递归比较仅/source_sha256不同，几何内容相同。哈希元数据有历史遗留，未擅自覆盖。
闭链runner初始化先构建PhysicalLqr求平衡并设置qpos/qvel/ctrl，随后full策略接管；
因此“无LQR”指策略运行中无反馈辅助，不应说初始化完全未用LQR工具。

复跑旧`wheel_policy_2900.onnx`到当前独立full runner：20000/20000步，passed=true，
final height .4114869m，max closure .0016665m，双轮contacts=[2,2]，无failure。
日志`old2900_closed_zero.log`与新9500的日志均在本次输出目录。
历史物理支撑成功可重现；同runner下9500在.111秒失接触，说明需要针对新策略的
初始化/映射适配行为排查，不是验证器或MuJoCo环境普遍不可运行。旧策略passed仍不
意味着零速度跟踪通过，历史约+.225m/s漂移限制依然适用。

旧记录：`/home/kellen/wheel_leg_mjrl-lqr/docs/sim2sim_validation.md`（2026-08-31），
接口文档`docs/fudan_sim2sim_contract.md`，原始日志`reports/sim2sim/`（均相对旧仓库）。
旧2900策略ONNX一致性、Isaac/MuJoCo tree稳态一致性通过；完整闭链策略零命令可支撑20s，
但零速滚动约+.225m/s，前后绝对跟踪失败。静态Phase B用了LQR，与无LQR策略验证区分。
因此存在成功的适配/物理支撑记录，不存在该报告证明的完整运动sim2sim成功。

当前独立验证目录：`/home/kellen/wheel_leg_sim2sim`，robot Conda环境。
未修改该仓库、原始参考目录或控制器。接口25D/125D/6D、100Hz policy，1kHz MuJoCo；
PD20/1，scale .5/10，原模型轮齿比15.7647。policy使用训练仓库已验收ONNX绝对路径。

asset_contract --check报stale contract。重新生成到/tmp对比，唯一差异/source/path：
`wheeled_infantry.xml`与`assets/wheeled_infantry.xml`。XML哈希相同
663e121ef9aed4bfab09ff0d3d98321b7ce77d80b63584235e4da691bd3aebf3；
nq45/nv44/nu6/njnt39/neq4。未覆盖原契约。

新结果：`plane/outputs/sim2sim_9500_20260921/`。
- closed_zero.log：full policy零命令，请求20000步，111步(.111s)被接触检查终止，
  wheels=(0,2)，无非轮触地。初始base约.2998m，终止时.3609m，最大closure1.954mm。
  是接触门槛失败，不等同已跌倒；2秒warmup未达到，均值字段0不能解释成真实稳态。
- tree_zero.log：同一训练URDF生成的parity XML，20000/20000步，interface_passed=true，
  高度范围.39943～.40089m，终值.39956m。仅零命令接口/运行验证，不是全速度验收。

复现命令（从训练仓库根目录）：
```bash
/home/kellen/anaconda3/envs/robot/bin/python /home/kellen/wheel_leg_sim2sim/sim2sim_closed_policy.py \
 --mode full --policy /home/kellen/fudan_train/plane/outputs/motion_goal_20260921_143444/accepted_basic_motion.onnx \
 --steps 20000 --forward 0 --yaw 0 --height .40 --metrics-warmup-seconds 2 --log-every 5000
/home/kellen/anaconda3/envs/robot/bin/python /home/kellen/wheel_leg_sim2sim/sim2sim_parity.py \
 --xml /home/kellen/fudan_train/assets/wheel_leg_train_parity.xml \
 --policy /home/kellen/fudan_train/plane/outputs/motion_goal_20260921_143444/accepted_basic_motion.onnx \
 --steps 20000 --forward 0 --yaw 0 --height .40 --log-every 5000
```

下一步核对闭链初始姿态/高度与policy reset分布、虚拟关节观测及接触时序，做有记录的
初始化对照；不移除接触门槛或加入LQR/配平来宣称策略成功。全闭链GUI/速度网格暂不称通过。
