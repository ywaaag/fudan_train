# 架构重构证据与交接

## 2026-09-24 阶段98：历史策略对比入口迁移

- `tools/compare_policy_versions.py`改为薄CLI，完整实现迁至
  `app.compare_policy_versions.main(root)`；候选、manifest、seed、randomization、缓存
  checkpoint/evaluator/asset hash及失败状态协议保持原值。
- robot环境--help通过，未运行历史评估。新增`test_compare_policy_entrypoint.py`，验证CLI
  无Popen、app导入不解析参数/读写job/启动任务。
- supervisor清单已将剩余直接子进程诊断收窄为continue_height_course、audit_observation_noise；
  wait_for_completion、summarize_policy_comparison、export_model_registry仍是旧job文件边界。

## 2026-09-24 阶段97：全仓库静态与测试收口复核

- 两仓库当前工作区除`.deep-copilot/`外无未提交源码改动；训练最新提交b3eb4ab/36147a4/
  221910e/c1a2f40，sim2sim最新提交ee84200，原dirty源码归档分支独立保存。
- 训练依赖审计：243模块、1027边，cycles/layer_violations/simulator_import_violations/
  missing_local_imports均为空；sim2sim：39模块、78边，cycles/layer_violations为空。
  生产源码扫描没有eval/exec/__import__动态执行（PyTorch `.eval()`调用除外）；tools与
  scripts没有互相导入旧监督器入口，测试/应用导入均指向责任模块。
- 最终全套测试：训练654 passed、2 warnings；sim2sim 24 passed。两套测试均在对应旧
  Python环境运行，未启动长训练、GUI、MuJoCo长验收或通知。本轮仅更新completion_audit与本记录。
- 已关闭静态循环、入口迁移和测试收口项；仍需完成supervisor_inventory中应用内部
  STOP/artifact/恢复协议的代表性复核，以及最终smoke/验收汇总和最后Git提交，goal保持active。

补充核对：3个历史诊断tools（compare_policy_versions、continue_height_course、
audit_observation_noise）仍直接维护旧子进程/报告协议，wait_for_completion、
summarize_policy_comparison、export_model_registry仍直接读取或写job文件。架构测试将
这些明确列为历史边界，其余已迁移入口禁止tools拥有Popen或status journal；不把允许列表当作
永久架构层，后续需按真实命令测试后再迁移。

## 2026-09-24 阶段96：GUI播放命令状态实例化

- 修改 `plane/wheel_legged_gym/scripts/play.py`：新增PlayCommandState，每次play会话独立
  持有命令、按键、running、RLock和panel limits；keyboard、web panel和仿真循环通过显式
  lambda/参数绑定访问。保持w/s/a/d/e/x/c行为、互斥转向、panel抢占、环境变量范围、
  jump ramp及相机参数不变；训练domain不读取GUI状态。
- 新增 `plane/tests/test_play_state_boundary.py`，静态验证状态字段、回调首参和模块级状态
  缺失；不启动Isaac、pynput或GUI。更新play_gui指南、ARCHITECTURE、completion_audit、
  依赖JSON及本记录。
- 652项基线加本轮测试后全套654 passed、2 warnings；静态243模块1027边，循环/层级/
  仿真导入/缺失本地模块均为0；未启动GUI、训练、仿真或通知。
- 当前剩余项收窄为第三方运行时边界、最终全仓库动态审计、代表性统一smoke/验收和Git
  收尾；goal仍保持active。

## 2026-09-24 阶段95：移除未使用的135D导出副作用并记录GUI边界

- 核对 `scripts/export_encoder_jit.py`：仓库无调用方，使用旧27D×5=135D输入、硬编码
  `/root/gpufree-data`路径，导入即读checkpoint/写JIT，与当前25D/125D/6D契约无关。
  按用户取消旧用法兼容要求删除运行文件，将原始字节精确归档到
  `docs/history/export_encoder_jit_135.py.txt`（长度1436、SHA256
  6bb9bddc765d7f98322a5c647063396a4ee0ce8e3f68e809ef70fd080bd87fde）；当前ONNX入口不变。
- AST扫描确认 `scripts/play.py` 的GUI模块级命令状态/RLock/runtime_limits仍是唯一明确的
 训练仓库共享可变交互状态缺口；新增 `docs/modules/play_gui.md`记录所有权、迁移计划和
 交互测试要求，不在无GUI测试情况下修改它。
- 更新compatibility、ARCHITECTURE、completion_audit、依赖JSON及本记录。全套652 passed、
  2 warnings；静态243模块1027边四类违规为空；未训练/启动GUI/通知。
  Goal仍未完成，下一项是GUI状态迁移或最终在有合适测试时明确其边界。

## 2026-09-24 阶段94：共享常量冻结与GUI全局状态缺口确认

- AST扫描包内模块顶层可变集合/调用，发现FUDAN_SCALES和TERMINAL仍可跨调用修改。
  修改experiments/recipes/fudan_stand.py为最终值的MappingProxyType，保持原字典插入顺序；
  workflows/completion.py的TERMINAL改frozenset。无奖励数值或完成状态集合内容变更。
- 新增plane/tests/test_readonly_runtime_constants.py，验证全部奖励键/值/顺序、manifest副本
  隔离与状态集合不可变；全套652 passed、2 warnings，244模块1027边四类违规为空。
- 同次扫描确认scripts/play.py仍有GUI全局命令/lock/runtime_limits，export_encoder_jit.py
  仍顶层加载模型并导出。更新ARCHITECTURE及completion_audit明确缺口，不以核心训练已分层
  推断整个仓库无全局状态。更新依赖JSON与本记录；未训练/修改物理/通知。

## 2026-09-24 阶段93：批量明确剩余监督器应用所有权

- 清单剩余13个tools流程入口改为薄CLI，实现迁至plane/wheel_legged_gym/app同名main(root)。
  完整名称见supervisor_inventory.md；保留main/辅助函数体，root及派生路径由调用参数构建，
  移除工具重复路径引导。没有批量改写实验、门槛、停止协议或日志格式。
- 新增plane/tests/test_remaining_supervisor_entries.py及fixtures/supervisor_entry_sources.json，
  26项检查冻结源码AST一致与导入无任务I/O。迁移test_legacy_yaw.py的audit_grid与gate导入，
  删除该测试的tools路径注入，不添加旧名兼容包装。
- 全套650 passed、2 warnings；8个有argparse入口在robot环境--help通过；无argparse入口未执行。
  静态244模块1027边四类违规为空。未训练/仿真/通知。
- 更新ARCHITECTURE、supervisor_inventory、completion_audit、依赖JSON及本记录。
  入口所有权完成不等于内部职责全部完成，剩余进程/artifact核对仍在台账中，goal保持active。

## 2026-09-24 阶段92：双姿态切换验收与应用解耦

- 修改plane/wheel_legged_gym/evaluation/height_acceptance.py增加assess_height_transitions，
  app/run_dual_height.py委托最终steady/response判断；保留原<=/.015/3秒与接触==0语义。
- 新增plane/tests/test_height_transition_acceptance.py（41项），与冻结原判断比较边界、
  None、NaN、基础拒绝、接触/未稳定及空集合；更新test_height_supervisor_entrypoints.py
  仅展开已独立验证的判断调用，剩余应用AST继续与原主体一致。
- 全套624 passed、2 warnings；231模块988边四类违规为空。更新height_supervisors指南、
  依赖JSON及本记录；不改训练/课程/物理参数，未启动训练或仿真。
  高度组进程/artifact边界和整体其他监督器仍待核对，goal未完成。

## 2026-09-24 阶段91：高度课程验收工作流

- 新增plane/wheel_legged_gym/workflows/height_evaluation.py，显式依赖EvaluationArtifacts/
  PythonJob端口及height_acceptance；app/run_height_course.py仅组装JobFiles/run并接收结果。
- 新增plane/tests/test_height_evaluation.py，5类场景对照冻结旧应用代码，检查命令、
  全部seed顺序、结果及JSON字节、异常不写acceptance；更新test_height_supervisor_entrypoints.py，
  仅展开已独立测试的工作流片段，剩余主函数AST仍与旧版精确比较。
- 全套583 passed、2 warnings；231模块988边四类违规为空。未启动训练/仿真/通知。
- 更新height_supervisors模块指南及依赖图JSON；本次减少应用验收细节，进程与升阶仍在app，
  整体goal及监督器清单剩余项尚未完成。

## 2026-09-24 阶段90：高度组应用入口与仿真导入解耦

- 修改tools/run_height_course.py、run_dual_height.py、screen_height_checkpoints.py为薄CLI；
  新增plane/wheel_legged_gym/app下三个同名模块，root显式传入，原main主体保留。
  移除两处无用Isaac导入及重复sys.path设置；配方和门槛只依赖既有experiments/evaluation。
- 新增plane/tests/test_height_supervisor_entrypoints.py及fixtures/height_supervisor_bodies.json，
  6项测试验证与c5bcfcb旧main主体AST一致和导入无任务I/O。
- 全套578 passed、2 warnings；robot环境height_course --help成功；静态230模块975边，
  四类违规为空。未启动高度训练、历史补筛或通知。
- 更新docs/modules/height_supervisors.md、supervisor_inventory.md、ARCHITECTURE、依赖JSON
  及本记录。进程/artifact仍明确待核对，不把迁移目录当作工作流全部完成。

## 2026-09-24 阶段89：提交当前重构工作树

- 用户明确要求Git提交。提交前两仓库index为空；逐项筛选当前tracked修改及未跟踪源码，
  只纳入.py/.md/.json，排除.deep-copilot、logs、outputs、checkpoint和缓存。
- 训练仓库提交标题`refactor: checkpoint readable training architecture`；sim2sim提交标题
  `refactor: expose explicit sim2sim validation boundaries`。均保留当前refactor分支，不push。
  当前状态包含原有用户改动，原始dirty版本另见archive/preexisting_20260923_100634，
  因此当前提交不能声称所有内容都由本轮重构新增。
- 提交前现场全套验证：训练572 passed、2 warnings，sim2sim 24 passed；两仓库静态循环
  与受管层级违规均为0（训练227模块966边，sim2sim39模块78边）。本轮无训练/通知。
- git diff --cached --check报告末尾多余空行及rewards/equations.py的一处行尾空白；
  本阶段保留冻结参考及当前源码，不混入批量格式清理。该格式检查未通过，不影响上述测试结论。
- 修改completion_audit.md及本记录说明阶段提交；具体源码文件清单使用git show --stat。
  监督器清单中的剩余项及最终验收尚未完成；本次仅保存当前成果，不标记goal完成。

## 2026-09-24 阶段88：监督器缺口具体化与高度门槛分离

- 新增 `evaluation/height_acceptance.py`（位于plane/wheel_legged_gym），修改
  `tools/run_height_course.py`通过assess_height_row执行原附加高度/接触门槛。
  新增 `plane/tests/test_height_acceptance.py`覆盖36组合；不改变micro/其他高度阈值、
  短路基础门槛、row.gate写入和失败原因顺序。
- AST盘点16个tools入口仍直接管理进程，新增docs/architecture/supervisor_inventory.md
  逐文件列出职责核查项与分组优先级；更新completion_audit.md和ARCHITECTURE入口。
  不将直接Popen等同于错误，不将已迁移app的文件重新算作待迁移。
- 全套572 passed、2 warnings；227模块966边四类违规为空；依赖JSON已更新。
  本轮未训练或启动历史监督器。整体goal尚有明确监督器组及最终Git/验收工作。

## 2026-09-24 阶段87：motion评估恢复边界

- 新增 `plane/wheel_legged_gym/app/motion_resume.py`，修改
  `plane/wheel_legged_gym/app/motion_supervisor.py`调用公开restore_evaluation_options。
  恢复journal读取、边界/checkpoint检查、PID探测、limits还原集中且顺序保持；
  STOP重命名、锁、日志及训练执行仍由应用后续原位置负责。
- 新增 `plane/tests/test_motion_resume.py`，10项验证正常/STOP恢复、训练状态拒绝、
  缺失checkpoint、存活supervisor/child和PermissionError；失败时选项不提前修改，
  读取不改journal或STOP、不启动进程。未增加原evaluating分支不存在的checkpoint预检。
- 全套536 passed、2 warnings；226模块963边四类违规为空；robot中motion --help成功。
  更新ARCHITECTURE、motion_supervisor指南、依赖JSON及本记录。未恢复训练或通知。
  剩余其他监督器/最终全范围验收/Git整理仍未完成，goal保持active。

## 2026-09-24 阶段86：将原始dirty源码建立独立Git归档

- 现场确认两仓库当前HEAD仍与20260923_100634备份HEAD相同，真实index没有暂存改动。
  使用临时GIT_INDEX_FILE重建备份diff/tar，以commit-tree/update-ref创建独立归档分支，
  不切换当前分支、不修改工作区或真实index。
- 训练仓库归档提交a53e7df142558beb037461bd4b94ce189e713b46，sim2sim归档提交
  0b53547451548de5bcf8fec8143f0a52d9e8957b；分支archive/preexisting_20260923_100634。
  分别校验60/1个归档源码文件blob，session日志留在原备份与目录，没有加入源码提交。
- 前后HEAD、status输出及index字节相同；证据写入
  plane/outputs/architecture_refactor_20260923/preexisting_git_snapshots.json。
- 新增 `docs/architecture/git_preservation.md`，更新completion_audit.md及本记录。
  无运行代码/依赖变化，不重复训练或测试。当前重构本身尚未提交，整体goal未完成。

## 2026-09-24 阶段85：固定高度诊断的嵌套进程边界

- 修改 `tools/run_fixed_height_diagnosis.py`为薄CLI，新增
  `plane/wheel_legged_gym/app/fixed_height_diagnosis.py`，root显式输入；实验、候选和seed
  顺序及状态/阈值保持。新增 `plane/wheel_legged_gym/adapters/processes/cooperative_job.py`，
  提取嵌套STOP传播与进程回收，通过三个回调报告状态，不导入应用或写status.json。
- 新增 `plane/tests/test_cooperative_job.py`，7种成功/退出失败/STOP/轮询异常/spawn失败
  场景验证回调、命令字面值和回收。与motion直接取消协议保持区分，没有强行合并。
- 更新 `docs/modules/fixed_height_diagnosis.md`、`ARCHITECTURE.md`、依赖JSON与本记录。
- 全套526 passed、2 warnings；最终仅格式调整后7项进程测试通过；225模块960边四类违规为空。
  没有执行该历史训练入口或--help（原脚本无argparse）；未训练/通知。
  剩余监督器与最终汇总/Git整理继续，goal未完成。

## 2026-09-24 阶段84：当前导出入口的ONNX最终对照

- 使用fudan_leg运行plane/export_onnx/export_onnx.py，参数log_root=plane/logs/wheel_legged、
  load_run=Sep22_11-19-04_motion_goal_20260922_111611_r01_basic_motion、checkpoint=10200，
  新输出为plane/outputs/architecture_refactor_20260923/final_reexport_10200.onnx。
- 新旧ONNX的SHA256均为4f3d974580eebf808bcd127381a403c8e890a9e927bbc92ab440674205a2950e；
  obs[batch,25]+obs_history[batch,125]→actions[batch,6]，名称、dtype及动态batch完全一致。
  CPU ONNXRuntime batch1/8/256逐元素精确一致，max_abs_error=0。
- 使用同目录verify_onnx.py --checkpoint=原10200.pt --onnx=新输出 --batch=256，
  PyTorch/ONNX max_abs_error=7.62939453e-06、mean_abs_error=6.95020105e-07，PASS。
  输出/日志/证据分别为final_reexport_10200.onnx、final_onnx_export.log、final_onnx_verify.log、
  final_onnx_equivalence.json，均在架构outputs目录；历史checkpoint和ONNX未修改。
- 本轮修改completion_audit.md及本记录，未改运行时代码或依赖图、未启动训练。
  此证据证明该checkpoint当前导出路径等价，不证明全部策略运动能力；整体goal仍有剩余项。

## 2026-09-24 阶段83：公开sim2sim正常/保护拒绝双路径复验

- 本轮不修改代码。robot解释器通过训练仓库tools/probe_closed_initialization.py公开进程
  接口复验两个场景，使用同一历史diagnostic_10200.onnx，hash保持
  4f3d974580eebf808bcd127381a403c8e890a9e927bbc92ab440674205a2950e。
- 正常：tree_zero、steps=2000、forward=1、metrics-warmup-seconds=.2、trace，
  输出`plane/outputs/architecture_refactor_20260923/final_closed_success.json`；passed=true，
  completed_steps=2000，对照policy_init_before.json。
- 拒绝：完整复用dynamic_boundary_20260922_124120/start_neg_ramp1p0_process.json的命令，
  仅改输出；tree_zero、sequence-json=start_neg_ramp1p0_sequence.json、steps=25000、
  metrics-warmup-seconds=5.0、trace。输出final_closed_rejection.json，
  passed=false、completed_steps=2489，腿长保护错误文本与历史结果完全相同。
- 两个JSON除runner_sha256和result.wall_seconds外全部字段（含trace/measurements）精确一致。
  输入/输出路径、SHA256和结果写入同目录closed_success_rejection_equivalence.json。
  进程退出码均为0，说明报告生成成功；不把退出0误解为保护拒绝案例通过运动验收。
- 外部仓库pytest 24 passed；其依赖39模块78边、cycles/layer_violations均为空。
  更新本记录及completion_audit.md，无依赖图变化、无长期训练、无保护或物理参数改动。
  其他监督器/最终汇总及Git整理仍未完成，goal保持active。

## 2026-09-24 阶段82：消除地形生成的eval依赖

- 生产源码扫描发现 `plane/wheel_legged_gym/adapters/isaacgym/terrain_generation.py`
  selected_terrain使用eval配置字符串；改为公开selected_generator显式列出10个原函数。
  不再支持任意Python表达式，未知名报ValueError；允许名称的函数身份及算法不变。
- 新增 `plane/tests/test_selected_terrain.py`（14项）：8个Isaac和2个本地函数身份、
  非函数表达式拒绝、实际pit布局/原type pop行为；修改 `plane/tests/test_architecture.py`
  阻止生产模块直接eval/exec/__import__，不把Torch的.eval()误报为动态代码。
- 更新 `docs/modules/terrain_creation.md`、`ARCHITECTURE.md`、依赖JSON及本记录。
  文档明确selected分支历史配置/属性前提没有顺手修复，不宣称该分支默认配置可用。
- 全套518 passed、2 warnings；随后新增的架构防回归7项通过（合计519项），
  静态223模块953边四类违规为空。未启动训练或改变地形参数。
  本次消除一项真实动态依赖；其他监督器和最终完整验收仍待完成，goal保持active。

## 2026-09-24 阶段81：TensorBoard摘要移除导入副作用

- 核对continue_stand_validated/train_stand_long调用hook的--job/--codex参数，
  当前app.completion仍支持，协议未失配；未执行通知。
- 修改 `tools/summarize_training.py`为CLI，新增
  `plane/wheel_legged_gym/app/training_summary.py`、
  `plane/wheel_legged_gym/adapters/artifacts/tensorboard_scalars.py`、
  `plane/wheel_legged_gym/evaluation/training_summary.py`；读取、统计、参数解析各自归属明确，
  导入不读取event或解析argv。新增 `plane/tests/test_training_summary.py`覆盖切片边界。
- 更新 `docs/modules/training_summary.md`、`docs/modules/historical_supervisors.md`、
  `ARCHITECTURE.md`、依赖JSON及本记录。依赖tools→app→adapter/evaluation，核心不依赖TB。
- 全套504 passed、2 warnings；223模块953边四类违规为空。
  真实已有asset_indices smoke日志用--window=100摘要，迁移前后stdout逐字节一致，证据
  `plane/outputs/architecture_refactor_20260923/training_summary_equivalence.json`。
  本轮未训练；其他监督器及最终完整验收仍待完成。

## 2026-09-24 阶段80：根架构导航与当前实现对齐

- 核对README、ARCHITECTURE、AGENTS、COMMANDS及模块/架构指南共25份文档，38个本地
  Markdown链接均存在；历史归档不按当前接口改写。
- 修改 `ARCHITECTURE.md`模块职责和主依赖图，补全环境协调、workflow/evaluation/ports、
  控制初始化、原点布局和Isaac资产索引入口；删除已完成边界仍标为未核对的旧说明。
- 修改 `README.md`，明确CLI保留与Python旧包装移除的区别；修改
  `docs/modules/compatibility.md`补全候选复核及四个历史监督器新入口。
- 对照源码确认ports三个公开协议名、跨仓库process_interfaces.json与外部VALIDATION_API.md
  的入口和测量快照契约一致。这里只验证入口文档，不扩大为所有CLI行为已验收。
- 本轮修改上述3份文档及本记录，未改运行时代码、依赖边或模型；不重复运行训练。
  旧监督器职责、最终运行验收及Git整理仍未完成，goal保持active。

## 2026-09-24 阶段79：历史站立/低速门槛独立验收

- 新增 `plane/wheel_legged_gym/evaluation/stand_continuation.py`及`low_speed_relay.py`，
  修改 `plane/wheel_legged_gym/app/stand_long_guard.py`和`run_low_speed.py`委托调用。
  接收已解析数据，不访问文件/进程，保留门槛、短路求值和拒绝原因顺序。
- 新增 `plane/tests/test_stand_continuation.py`、`test_low_speed_relay.py`，共40项与
  冻结旧监督器fixture对照；修改 `test_supervisor_entrypoints.py`只归一化提取的门槛调用，
  继续核对剩余启动/执行主体。旧fixture未改动。
- 全套498 passed、2 warnings；最后仅格式整理后针对48项门槛/入口测试再次通过。
  静态220模块943边、四类违规为空。未启动训练/仿真/通知。
- 更新 `docs/modules/historical_supervisors.md`、`ARCHITECTURE.md`、依赖JSON和本记录，
  明确low_speed空报告all([])沿用旧语义，不能把独立函数返回True当作样本完整性证明。
  其他监督器/最终验收仍未完成，goal保持active。

## 2026-09-24 阶段78：消除四个监督器的导入即执行

- 修改tools下`audit_stand_push.py`、`continue_stand_validated.py`、`run_low_speed.py`、
  `stand_long_guard.py`为薄CLI，新增app下四个同名模块；main(root)持有各任务state/闭包，
  导入不再解析参数、创建job、读模型或启动训练。原命令和执行顺序保留。
- 新增 `plane/tests/test_supervisor_entrypoints.py`及
  `plane/tests/fixtures/legacy_supervisor_bodies.json`：冻结迁移前主体，排除root注入后AST
  完全相等；导入测试禁止parse_args/mkdir/read_text/write_text/Popen。
- 对照测试捕获stand_long_guard分号同行语句重复提取，已修正；最初AST hash跨Python版本
  不一致，改用同解释器解析冻结旧主体进行结构比较，不重建基线为新实现。
- 更新 `docs/modules/historical_supervisors.md`、`ARCHITECTURE.md`、依赖JSON与本记录。
  明确历史watcher协议及门槛/进程抽取仍须核对，不以迁目录代替完整工作流模块化。
- 全套458 passed、2 warnings；218模块937边，四类违规为空。三个有argparse的CLI
  在robot环境--help成功；audit_stand_push无help，不执行。没有运行训练/仿真/通知。
  其他监督器及最终全范围验收未完，goal保持active。

## 2026-09-24 阶段77：闭链候选复核的入口、门槛与报告分层

- `tools/run_candidate_closed_review.py`改为薄CLI；应用实现迁至
  `plane/wheel_legged_gym/app/candidate_closed_review.py`，root/argv显式输入。
- 新增 `plane/wheel_legged_gym/evaluation/closed_precheck.py`存放原yaw预检门槛；
  新增 `plane/wheel_legged_gym/workflows/closed_review_report.py`生成原Markdown。
  依赖tools→app→evaluation/workflows，无反向导入。锁、STOP、子job发现、通知顺序未改。
- 新增 `plane/tests/test_closed_precheck.py`，8项覆盖全部拒绝门槛/hash变化及应用预检失败
  停止路径，验证PID、报告、仅一次子进程调用及不通知；没有真实启动MuJoCo或训练。
- 更新 `docs/modules/candidate_closed_review.md`、`ARCHITECTURE.md`、依赖JSON及本记录。
- 全套pytest 450 passed、2 warnings；214模块925边、四类违规为空；robot环境CLI --help
  通过。其他监督器和最终全范围验收仍待完成，goal保持active。

## 2026-09-24 阶段76：资产索引与环境创建协调分离

- 新增 `plane/wheel_legged_gym/adapters/isaacgym/asset_indices.py`，以显式Gym/actor、
  名称、配置和已有索引输入构造AssetIndices；修改
  `plane/wheel_legged_gym/envs/base/legged_robot.py`逐字段接收，不注入动态属性。
- 新增 `plane/tests/test_asset_indices.py`及冻结
  `plane/tests/fixtures/asset_indices_reference.py`，64组正常/缺失资产对照，保留错误文本、
  查询顺序、重复接触名、body handle区别、既有bilateral索引及feet clone语义。
- 首次全套测试发现新增测试先导入Torch导致旧Isaac导入顺序错误，已在测试入口先导入Isaac，
  运行时代码未改导入协议；重跑442 passed、2 warnings。
- 更新 `docs/modules/asset_indices.md`、`docs/modules/environment_lifecycle.md`、
  `ARCHITECTURE.md`、`docs/architecture/completion_audit.md`、依赖JSON及本记录。
  211模块916边，循环、层级、核心仿真导入、缺失本地模块均为0。
- 64env/seed11/1iteration/run_name=architecture_asset_indices_20260924 smoke退出0；
  `Sep24_16-06-01_architecture_asset_indices_20260924/model_1.pt`与原reward smoke的模型、
  两组optimizer及iter精确一致。证据为outputs/architecture_refactor_20260923下
  asset_indices_smoke.log及asset_indices_smoke_equivalence.json。
  环境已识别的拆分缺口处理完毕，仍需整体监督器整理与最终全范围验收，goal未完成。

## 2026-09-24 阶段75：环境原点布局独立计算

- 新增 `plane/wheel_legged_gym/domain/geometry/origins.py`，以OriginLayout显式返回
  原点、地形等级/类别、分桶索引和边界；修改 `plane/wheel_legged_gym/envs/base/legged_robot.py`
  显式接收字段。plane分支不生成rough专用环境属性，不用动态setattr。
- 新增 `plane/tests/test_origin_layout.py`、`plane/tests/fixtures/origin_layout_reference.py`，
  24组旧实现精确对照覆盖四种mesh、课程开关、单环境和非方形环境数、dtype及RNG。
- 更新 `docs/modules/origin_layout.md`、`docs/modules/environment_lifecycle.md`、
  `ARCHITECTURE.md`、`docs/architecture/completion_audit.md`、依赖JSON与本记录。
- 全套pytest 378 passed、2 warnings；210模块912边，四类违规为空。
  64env/seed11/1iteration/run_name=architecture_origins_20260924 smoke成功；
  `Sep24_15-59-16_architecture_origins_20260924/model_1.pt`对照原reward smoke的模型、
  两组optimizer及iter精确一致，证据为outputs/architecture_refactor_20260923下
  origins_smoke.log及origins_smoke_equivalence.json。未启动长训练。
- 环境剩余资产索引组装及整体监督器/最终验收仍需完成，goal保持active。

## 2026-09-24 阶段74：默认关节姿态与控制初始随机化

- 新增 `plane/wheel_legged_gym/domain/control/initialization.py`，显式输入配置、buffer和
  采样/report回调，返回InitialControlState；修改
  `plane/wheel_legged_gym/envs/base/legged_robot.py`通过公开入口组装。
- 新增 `plane/tests/test_control_initialization.py` 与
  `plane/tests/fixtures/control_initialization_reference.py`；对照提取前代码全部32种开关组合，
  验证多重匹配最后覆盖、未匹配提示、五类随机化调用顺序、RNG、参数值及delay存储身份。
- 更新 `docs/modules/control_initialization.md`、`docs/modules/environment_lifecycle.md`、
  `ARCHITECTURE.md`、`docs/architecture/completion_audit.md`、依赖JSON及本记录。
  依赖为envs→domain.control，采样通过显式回调，domain无Isaac依赖。
- 验证：fudan_leg全套pytest 354 passed、1 warning；静态209模块908边，四类违规为空。
  64env/seed11/max_iterations=1/run_name=architecture_control_init_20260924真实smoke成功；
  `Sep24_15-55-59_architecture_control_init_20260924/model_1.pt`与原reward smoke的模型、
  optimizer、extra_optimizer、iter全部精确一致。证据在
  `plane/outputs/architecture_refactor_20260923/control_init_smoke.log`及
  `control_init_smoke_equivalence.json`。没有长期训练，goal仍待全范围验收。

## 2026-09-24 阶段73：生命周期完整阅读路径与剩余职责定位

- 现场逐段核对LeggedRobot构造、step、post_physics_step、reset_idx、命令callback、
  buffer初始化及BaseTask.reset，新增 `docs/modules/environment_lifecycle.md`。
- 文档包含初始化顺序、状态所有权表、step模块图、reset顺序及method/legacy不同指标时序。
  明确action_fifo不会在reset清空、重采样后mode用于采样比例、有限差分可替换dof_vel视图，
  防止后续重构以“清理”为由改变既有行为。
- 更新 `ARCHITECTURE.md`入口和 `docs/architecture/completion_audit.md`。
  将笼统的生命周期待拆分改为源码可定位的剩余职责：默认DOF/PD初始化随机化、
  资产索引和地形初始位置组装。跨模块调度仍明确归环境协调层，不制造代理层。
- 本轮只修改上述3份文档及本记录，无运行时代码或依赖变化；文档对应方法已核对，
  未重复运行训练/测试。完整goal未完成，阶段71的数值证据范围不扩大为全配置保证。

## 2026-09-24 阶段72：删除无调用方的私有兼容别名

- 扫描生产源码、tools及tests，确认四个旧名字只剩定义；删除_gym_args、
  _disable_randomization、_set_fixed_ranges、_apply_method_randomization全部9处别名。
- 修改文件：`plane/wheel_legged_gym/adapters/isaacgym/evaluation_setup.py`、
  `plane/wheel_legged_gym/adapters/isaacgym/policy_io.py`、
  `plane/wheel_legged_gym/scripts/isaac_parity_trace.py`、
  `plane/wheel_legged_gym/scripts/isaac_command_grid.py`、
  `plane/wheel_legged_gym/experiments/primitives.py`。
- 更新 `docs/modules/configuration.md`、`ARCHITECTURE.md`和
  `docs/architecture/completion_audit.md`，纠正已完成的actor、Torch设置、guide配置、
  指标所有权和motion轮次仍被写为未完成的旧说明；更新依赖JSON和本记录。
- 公开函数体及调用方不变，不为无实际使用的历史接口维持两套名字。
  全套322 passed、1 warning；208模块904边、四类违规为0，旧别名源码搜索无命中。
  本阶段未训练；剩余生命周期/其他监督器/最终验收和Git整理仍未完成。

## 2026-09-24 阶段71：命令指标单一所有权与reset汇总

- 修改 `plane/wheel_legged_gym/envs/base/legged_robot.py`，删除25个command_metric_*别名及
  单独buffer列表，统一使用command_metrics；指标绑定冻结，累计显式写入张量切片。
- 修改 `plane/wheel_legged_gym/domain/commands/tracking_metrics.py`，增加summarize；
  将reset中的18项日志聚合移入指标模块，环境仍保持汇总与课程/物理reset的原顺序。
- 同步迁移 `plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py` 的饱和度读取。
- 新增冻结 `plane/tests/fixtures/metric_summary_reference.py`；修改
  `plane/tests/test_tracking_metrics_equivalence.py`，覆盖零/非零分母与不改buffer；
  修改 `plane/tests/test_control_observation_equivalence.py`，迁移测试替身的指标接口，保留旧基线。
- 更新 `docs/modules/simulation_tensors.md`、依赖JSON与本记录。依赖方向未改变，
  静态208模块904边，四类违规均为0。全套322 passed、1 warning。
- 64env/seed11/1iteration smoke成功，`Sep24_09-36-43_architecture_metric_owner_20260924/model_1.pt`
  与原基线模型、两组optimizer、iter精确相同；证据为
  `plane/outputs/architecture_refactor_20260923/metric_owner_smoke_equivalence.json`及同名日志。
  无长训练；其他buffer与最终全范围验收仍待完成。

## 2026-09-24 阶段70：命令重采样与环境分离

- 新增 `plane/wheel_legged_gym/domain/commands/resampling.py`，显式CommandBuffers、ranges、
  config、device和两项回调；修改 `plane/wheel_legged_gym/envs/base/legged_robot.py` 委托调用。
  采样不再依赖完整环境对象，删除该方法内未执行的历史jump注释，环境职责缩减约100行。
- 新增 `plane/tests/test_command_resampling.py` 与冻结旧方法
  `plane/tests/fixtures/resampling_reference.py`；20组策略/heading/动态组合精确比较buffer、
  RNG状态及reset调用时刻，另验证未知策略异常。
- 发现并修正提取时设备类型问题：Isaac heading TorchScript函数要求str，保留显式原device。
- 320项全套测试通过；64env、seed11、1iteration实际smoke退出0。
  `Sep24_09-33-33_architecture_resampling_20260924/model_1.pt`对照原reward smoke，
  模型、optimizer、extra_optimizer、iter全部精确相同。日志及JSON在
  `plane/outputs/architecture_refactor_20260923/resampling_smoke*`。
- 更新 `docs/modules/command_resampling.md`、`ARCHITECTURE.md`、
  `docs/architecture/completion_audit.md`、依赖JSON与本记录。依赖208模块904边，四类违规为0。
  未恢复长训练。下一项为buffer及生命周期协调核对，整体goal未完成。

## 2026-09-24 阶段69：禁止反向导入可执行入口

- 核对依赖JSON中所有tools/scripts的实际边；未发现入口之间或核心到tools/export的现存导入。
- 修改 `tools/check_architecture.py`，禁止任何模块导入tools、export_onnx或
  wheel_legged_gym.scripts可执行模块，补上过去仅检查包内层级的盲区。
- 修改 `plane/tests/test_architecture.py`，覆盖app→tools、tools→tools、adapter→export、
  scripts→scripts拒绝，以及tools→app合法方向。
- 更新 `ARCHITECTURE.md`、`docs/architecture/dependencies.json`及本记录。
- 架构测试6 passed；依赖207模块900边，四类违规清单均为空。规则不改变运行行为，
  不代表旧监督器内部职责已经全部完成整理；goal保持active。

## 2026-09-24 阶段68：候选验收后的轮次推进

- 新增 `plane/wheel_legged_gym/workflows/motion_round.py`：RoundProgress/Mode显式传参，
  advance_round负责候选动态门槛、导出后接受、source更新、升阶和focus；无全局状态。
- 修改 `plane/wheel_legged_gym/app/motion_supervisor.py` 调用工作流，应用仍拥有状态保存、
  训练进程、停滞上限和异常处理。依赖为app→motion_round→candidate_decision/validation。
- 新增 `plane/tests/test_motion_round.py`，16项覆盖拒绝/改善/升阶和故障顺序；
  导出失败不记录接受，进入下一阶段评估失败保留已导出的接受记录；最终课程通过仍等待独立验证。
- 更新 `docs/modules/motion_supervisor.md`、`ARCHITECTURE.md`、
  `docs/architecture/dependencies.json` 与本记录。
- 验证：fudan_leg全套pytest 299 passed，1 warning；依赖207模块900边，循环、
  层级违规、核心仿真导入和缺失本地模块均为0；robot环境motion CLI --help通过。
  没有启动训练、修改物理或修改MuJoCo。完整goal仍需按completion_audit收尾。

## 2026-09-24 阶段67：训练计划与监督器执行分离

- 新增 `plane/wheel_legged_gym/workflows/motion_training_plan.py`，公开冻结选项、spec、
  去重签名和恢复训练参数构造；修改 `plane/wheel_legged_gym/app/motion_supervisor.py` 调用它。
  应用仍拥有文件、CLI验证、进程和状态；workflow不反向依赖应用或读取外部状态。
- 新增 `plane/tests/test_motion_training_plan.py` 与
  `plane/tests/fixtures/motion_training_plan_reference.py`：冻结提取前逻辑，对照36组配置，
  覆盖默认字段省略、动态开关、零比例对照、focus重复顺序及签名；另验证恢复命令与seed身份。
- 更新 `ARCHITECTURE.md`、`docs/modules/motion_supervisor.md`、
  `docs/architecture/completion_audit.md`、`docs/architecture/dependencies.json`。
  明确MuJoCo作为独立验证工具使用，后续不继续全面拆分其内部实现。
- 验证：fudan_leg中 `python -m pytest plane/tests -q --disable-warnings --maxfail=1`：
  283 passed，1 warning；`python tools/check_architecture.py --write docs/architecture/dependencies.json`：
  206模块897边，cycles/layer_violations/simulator_import_violations/missing_local_imports均为空。
  robot环境中 `python tools/run_motion_goal.py --help` 成功；未启动训练或修改sim2sim。
- 收益：检查某轮将使用何种实验配置只需读训练计划模块，无需加载监督器的锁、通知和子进程流程。
  轮次状态推进及其他监督器仍是未完成任务，goal保持active。

## 2026-09-24 阶段66：测量接口只保留公开模块调用

- 按用户最新要求，不保留无调用方的旧用法兼容层。
- 修改 `/home/kellen/wheel_leg_sim2sim/sim2sim_closed_policy.py`：删除私有测量别名，
  统一 `policy_measurements.function(...)`；避免函数与局部测量结果同名遮蔽。
- 更新外部仓库 `dependencies.json`；依赖仍为 runner → policy_measurements，
  39模块/78边，循环及受管层级违规均为0。
- 外部仓库 `python -m pytest tests -q`：24 passed。
- 真实2000步closed probe输出 `plane/outputs/architecture_refactor_20260923/public_measurements_qualified.json`，
  与 `snapshot_after.json` 比较，仅排除运行耗时和runner源码hash，所有其余字段完全一致。
- 更新 `docs/architecture/completion_audit.md`，移除已过时的旧CLI兼容验收要求，
  改为单一公开入口与实际调用方迁移；整体架构goal尚未完成。

## 2026-09-23 第一阶段

- 两仓库分支：`refactor/codex-readable-20260923_100634`。
- 原始dirty源码/未跟踪文件、HEAD及diff备份：`/home/kellen/refactor_backups/20260923_100634`。
- 没有清理用户文件、日志或模型；`.deep-copilot/`保持原样。

### 已执行

1. 实验共同原语与选择器分离；具体配方迁入experiments/recipes，回环消除。
2. envs/utils包不再隐式注册/构造训练栈；CLI显式创建TaskRegistry实例。
3. 配置下沉contracts；网络/PPO/storage/runner迁入learning；旧路径是兼容包装。
4. 网络类和算法选择改为明确的允许映射，移除eval。
5. 评估共用的参数、随机化和策略加载函数不再从其他CLI导入，迁入Isaac适配器。
6. 正式闭链probe改成跨进程公开入口；sim2sim runner显式接收初始化和after_step回调，
   删除该probe的PhysicalLqr/mj_step全局替换。测量顺序仍为step后、保护前。
7. 噪声诊断改用显式diagnostic-mode参数，删除评估器/策略函数的全局替换。
8. 新增可重复静态依赖扫描，包含包初始化和函数内导入，目前188模块、712条边、0循环组。

### 已验证

- 重构前：89项训练仓库tests通过。
- 配方/网络/配置迁移后：同样89项通过。
- 旧train.py入口，64环境、seed11、1iteration实际运行成功，3072 timesteps；仅架构smoke。
- 原10200 ONNX、tree_zero、500个1ms物理步、.2秒warmup、trace：重构前后结果字段
  （除wall_seconds）、完整trace、measurements逐项完全相同。
- 对照原始JSON：`/tmp/refactor_closed_before.json`、`/tmp/refactor_closed_after.json`。
- 阶段最终回归：训练仓库91项通过（新增两项架构检查），sim2sim仓库2项通过。
- 所有上述日志与前后JSON已复制到仓库`plane/outputs/architecture_refactor_20260923/`，
  `verification.json`保存比较结果和证据SHA，后续不依赖/tmp存在。

完整目标尚未结束。后续任务见根ARCHITECTURE的“未完成迁移”，禁止根据本页宣称所有层级
依赖已清零或完整高速sim2sim回归已通过。所有下一步均需保持可运行及旧契约兼容。

## 第二阶段：奖励计算与仿真生命周期解耦

- 修改模块：`domain/rewards/{api,inputs,equations}`、`domain/geometry/rotations`及环境委托。
- 环境原54个奖励/辅助公式迁至独立模块，公开入口`evaluate(name, RewardInputs)`。
- 依赖变化：环境→reward API→纯张量公式/几何；奖励不再隐含依赖整个LeggedRobot。
- 显式37字段冻结输入结构，无mixin或运行时动态挂方法，原奖励调用名称/顺序不变。
- 新增`docs/modules/rewards.md`描述最小阅读路径、输入所有权及测试入口。
- 前后四种配置分支、54公式共216项逐张量结果完全相等；测试同时保证输入张量不变。
  黄金数据保存于`plane/tests/fixtures/reward_equivalence.json`，由重构前实现生成，不自动刷新。
- robot环境仅安装PyTorch即可运行该奖励测试，不加载Isaac Gym。
- 64env/seed11/1iteration实际smoke前后模型、两组optimizer、iteration均精确相等，
  证据为`plane/outputs/architecture_refactor_20260923/reward_smoke_equivalence.json`。
- 总测试92项通过；依赖审计193模块、741条边、0循环组。

本阶段不是奖励优化。物理初始化/terrain/课程统计、workflows迁移及完整层级禁止规则仍待完成。

## 第三阶段：验收、命令与通知边界

- 命令ramp/sequence迁入domain/commands，动态指标迁入evaluation，gate不再位于可执行比较脚本。
- 并行验证调度迁入workflows，reference smoke核验迁入artifacts适配器。
- 完成hook拆为workflows报告、notifications传输、app组装及NotificationSink接口；
  报告工作流不导入HAPI，消息ID、锁、确认记录和失败不重试语义不变。
- tools调用方改为明确的模块导入，旧命令/函数路径保留薄包装。CLI bootstrap限定在tools入口，
  核心模块不改变sys.path。噪声诊断等旧CLI仍可单独启动。
- 新增evaluation/completion最小模块指南。依赖检查增加迁移层的允许边与核心禁止simulator导入。
- 新图当前204模块、836边，循环和已迁移层级违规均为0。未迁移envs/tools不借此宣称已完成分层。
- 完成hook和响应汇总旧CLI的`--help`实际运行成功，没有发送消息或启动训练。
- 阶段最终92项tests通过。用备份中原始汇总代码和两份实际历史rollout分别计算，
  动态响应与闭链序列的输出逐字段完全一致，证据为
  `plane/outputs/architecture_refactor_20260923/evaluation_equivalence.json`。

## 第四阶段：启动工具与源码溯源

- 原`utils/helpers.py`的函数体原样拆到6个责任文件：
  `contracts/config_serialization.py`、`app/arguments.py`、`app/random_seed.py`、
  `adapters/isaacgym/simulation_parameters.py`、`adapters/artifacts/checkpoints.py`、
  `adapters/artifacts/jit_export.py`。旧helpers保留显式兼容导出。
- 环境和评估只引用配置转换；不再因转换一个配置而导入CLI和Isaac参数构造。
  registry与play/train入口直接导入各责任模块，函数内部运算和随机数调用顺序未改。
- 修复旧`rsl_rl/modules/api.py`兼容入口遗漏的网络构造器导出。
- 新增`adapters/artifacts/source_snapshot.py`及其回归测试：run保存保留旧顶层快照，
  同时增补包内Python源码相对路径树及SHA256。否则模块迁移后旧快照只能记录兼容包装，
  无法说明实际执行了什么。历史run不回填，已有快照拒绝覆盖。
- `app/task_registry.py`负责调用快照；新增`docs/modules/configuration.md`说明入口及最小阅读路径。
- 验证命令：`python tools/check_architecture.py --write docs/architecture/dependencies.json`；
  使用fudan_leg解释器运行`python -m pytest plane/tests -q --disable-warnings --maxfail=1`。
- 当前211模块、903依赖边；静态循环、已迁移层违规及核心仿真器导入违规均为0。
  测试93项通过。仍未完成环境生命周期、实验监督器及剩余跨仓库诊断入口的重构。

## 第五阶段：环境控制与观测边界

- 新增`domain/control/{__init__,actuation}.py`和`domain/observations/{__init__,policy}.py`；
  修改`envs/base/legged_robot.py`为显式传参调用。PD、25D观测、noise幅值和history操作
  可独立阅读，核心计算不需要访问整个环境对象或导入Isaac Gym。
- saturation统计和clipping留在环境调用处；noise仍在privileged组装后采样；
  历史slice和更新条件均保留，不顺手修改原有语义。
- 新增`test_control_observation_equivalence.py`和冻结的
  `fixtures/control_observation_reference.py`；16组组合逐张量及RNG状态精确一致。
- 新增`docs/modules/policy_io.md`，同步根ARCHITECTURE模块图和生成的dependencies.json。
- 全套109项测试通过；依赖图215模块、911边，静态循环及已迁移层违规均为0。
- 实际64env/seed11/1iteration smoke完成，模型、主optimizer、extra optimizer、iter
  与上一阶段checkpoint精确相等；新增source_snapshot的163份源码hash全部验证通过。
  证据：`plane/outputs/architecture_refactor_20260923/control_smoke_equivalence.json`、
  `control_smoke.log`、`control_tests.log`。
- privileged观测、生命周期/物理构建/课程状态及监督器等仍待迁移，整个goal未完成。

## 第六阶段：Motion/height实验输入显式化

- 修改`experiments/recipes/{motion_goal,height_course}.py`和`experiments/selection.py`：
  配方显式接收spec，不再读取环境变量/JSON或checkpoint。
- 新增`app/experiment_inputs.py`解析旧环境输入，新增`adapters/artifacts/recipe_source.py`
  持有原有来源校验。依赖为app→recipe/adapter，没有recipe→app或文件读取的隐式依赖。
- 修改旧`envs/wheel_legged/{policy_experiments,motion_goal,height_course}.py`兼容入口；
  修改`scripts/train.py`的来源校验导入，并修复此前未被默认smoke覆盖的resume分支中
  对已移除utils导出的引用，统一直达artifacts/checkpoints。
- 环境输入相关既有测试改为验证app边界：`test_dynamic_equivariance.py`、
  `test_height_course.py`、`test_motion_geometry.py`、`test_motion_switch_recipe.py`。
  新增`test_experiment_inputs.py`，覆盖显式配置不依赖ambient env、旧入口等价及拒绝条件。
- 全套115项测试通过；静态图217模块、946边，循环与已迁移层违规均为0。
- 直接执行备份tar中的旧motion实现、Git原始height实现，与新配方做4组配置/optimizer/
  manifest精确对照，全部相等；证据为`recipe_equivalence.json`，包含旧源码SHA256。
- 更新本页、根ARCHITECTURE、dependencies.json及`docs/modules/experiment_inputs.md`。
  未启动训练；stand环境输入等剩余项仍需继续迁移。

## 第七阶段：监督器决策和进程所有权

- 从`tools/run_motion_goal.py`抽取`assess`、`geometry_retained`至
  `evaluation/motion_candidates.py`，抽取子进程生命周期至
  `adapters/processes/{__init__,python_job}.py`。监督器显式注入3个状态回调，
  adapter不接收整个state、不负责通知或挑选模型，保持STOP和保存顺序。
- 删除监督器不需要的Isaac导入，旧`--help`使用robot解释器运行成功。
- `test_motion_goal.py`和`test_motion_geometry.py`直接引用评价模块。
  新增`test_python_job_runner.py`覆盖真实短进程的5类退出/取消路径；全套120项通过。
- 用备份中的原始评分实现对真实历史rollout及3组geometry情形做对照，完全一致；
  证据在`plane/outputs/architecture_refactor_20260923/motion_assessment_equivalence.json`。
- 依赖审计220模块、953边，循环及已迁移层违规为0。更新根架构图、依赖JSON及
  `docs/modules/motion_supervisor.md`，包含原有故障路径边界，避免将未修复行为误述为保障。
- 轮次工作流与持久化仍在tools入口，整个架构goal尚未完成；本轮没有启动RL训练。

## 第八阶段：状态持久化与CLI组装

- 新增`adapters/artifacts/job_files.py`、`workflows/motion_report.py`及
  `app/motion_status.py`，分别负责文件写入、纯报告文本、存储/通知顺序组装。
  核心报告不读取环境变量、不直接写文件或通知；这些依赖由app显式注入。
- `tools/run_motion_goal.py`变为薄兼容入口；应用移至`app/motion_supervisor.py`，
  根目录作为参数传入，模块不再修改sys.path或持有仓库路径全局变量。
  参数定义分至`app/motion_options.py`，默认值/选项保持不变。
- 新增`test_motion_status.py`覆盖终态/session组合、报告失败、wake失败及保存顺序。
  全套126项测试通过，旧CLI重构前后help字节一致。
- 对备份中的原始报告构造代码与真实历史status生成文本逐字对照，一致；证据在
  `plane/outputs/architecture_refactor_20260923/motion_report_equivalence.json`。
- 静态图225模块、975边，循环及已迁移层违规为0；git diff --check通过。
  更新监督器模块指南和依赖JSON。app中的轮次/evaluate/export工作流仍待继续拆分，
  本轮未启动训练或发送通知，goal未完成。

## 第九阶段：清除配方环境变量依赖

- 修改`experiments/primitives.py`、`selection.py`、`recipes/stand_balance.py`：
  stand随机化以显式参数传入，实验目录不再读取任何环境变量。
- `app/experiment_inputs.py`解析旧环境变量；更新兼容入口
  `envs/wheel_legged/{policy_experiments,stand_balance}.py`及评估CLI
  `scripts/{evaluate_standing,evaluate_policy_comparison}.py`，保留旧调用结果。
- 扩展`test_experiment_inputs.py`验证四种level、错误ambient输入不影响显式调用、
  非stand忽略override及实验目录不读环境变量的静态边界。
- 全套132项测试通过；8组原实现/新实现的配置、optimizer配置、manifest完全一致，
  证据`plane/outputs/architecture_refactor_20260923/stand_input_equivalence.json`。
- 更新根架构文档、输入模块指南和依赖JSON：225模块、988边，循环/已迁移层违规为0。
  未训练；legacy来源文件读取、环境生命周期及剩余监督器工作流仍待完成。

## 第十阶段：Legacy来源与checkpoint迁移边界

- 新增`adapters/artifacts/legacy_sources.py`，迁入stop_retention、legacy_anchors、
  legacy_speed2、legacy_speed2_stop、legacy_yaw、h3_low_speed、h3_speed1的来源校验。
  新增`checkpoint_migration.py`，迁入verify_full_resume/verify_and_restore_std。
- 上述7个recipe删除文件读取职责；clean_observation删除未使用的校验再导出。
  对应7个envs兼容包装保留旧函数名；scripts/train.py和2个H3测试改为直接导入artifact入口。
- 校验9项函数AST与备份/Git原始实现一致（只允许validate_source函数重命名），
  包括拒绝条件、检查顺序和迁移操作；证据路径仍解析到同一根目录。记录为
  `plane/outputs/architecture_refactor_20260923/legacy_source_equivalence.json`。
- 扩展`test_experiment_inputs.py`阻止配方重新引入文件操作；check_architecture将
  experiments纳入层级约束，仅允许依赖contracts/domain/experiments。
- 更新ARCHITECTURE、实验模块指南和依赖JSON：227模块、1020边，循环与迁移层违规为0。
  全套133项测试通过；未训练，环境生命周期与跨仓库/监督器剩余任务仍待完成。

## 第十一阶段：Critic观测与地形创建

- 新增`domain/observations/critic.py`，从LeggedRobot抽取privileged observation；
  显式输入代替环境对象访问，保留batch mass均值、height clip及actor噪声前组装顺序。
  更新`test_control_observation_equivalence.py`的实际方法执行依赖，16组冻结旧实现对照通过。
- 新增`adapters/isaacgym/terrain_creation.py`，地面/heightfield/trimesh创建参数构造
  从环境迁入适配器；环境方法仅传入gym/sim/material/terrain/device并接收height_samples。
- 新增`test_terrain_creation_equivalence.py`和冻结fixture
  `fixtures/terrain_creation_reference.py`，以真实Gym参数类比较3类旧新调用参数、数组和张量。
- 修改`envs/base/legged_robot.py`，同步policy_io指南、新增terrain_creation指南、
  根ARCHITECTURE与依赖JSON。全套136项测试通过，循环和迁移层违规为0。
- 未启动训练。本次测试证明计算/参数对照等价，不替代最后的真实仿真验收。
  actor创建、terrain数据生成和环境课程生命周期仍待迁移。

## 第十二阶段：课程checkpoint边界

- 新增`domain/commands/checkpoint_state.py`，拆出method/staged payload、counter恢复
  和stage校验；修改`envs/base/legged_robot.py`显式委托，保留课程分支和恢复赋值顺序。
- 不新增checkpoint字段，不持久化旧实现未保存的window/last_check_step；不改变错误形状
  counter的忽略行为。窗口重置和范围应用仍由环境显式执行。
- 新增`plane/tests/test_course_checkpoint.py`六项兼容测试及
  `docs/modules/course_checkpoint.md`；同步根架构入口和生成依赖图。
- 全套142项测试通过，静态图230模块/1032边，循环与迁移层违规为0；未训练。
  该阶段只完成持久化边界，课程推进/累计和环境其他生命周期仍待继续拆分。

## 第十三阶段：课程推进决策

- 新增`domain/commands/staged_progress.py`，环境中的检查时机、升阶决策和日志字段
  改为调用显式函数；修改`envs/base/legged_robot.py`保留stats收集及范围/窗口副作用。
- 新增`test_staged_progress_equivalence.py`和冻结的
  `fixtures/staged_progress_reference.py`，六种有效课程边界的stage/streak、日志指标、
  last_check_step及范围更新/窗口重置顺序与旧方法一致。
- 全套148项测试通过；依赖图231模块/1036边，静态循环与迁移层违规为0。
  更新课程模块指南和dependencies.json；未启动训练。窗口累计及其余生命周期仍待完成。

## 第十四阶段：Mapping audit跨仓库所有权

- 将`tools/audit_closed_mapping.py`实现原样迁至sim2sim仓库`audit_mapping.py`，删除跨仓库
  sys.path修改；旧tools入口仅解析仓库位置并调用公开进程接口。
- `adapters/mujoco/api.py`新增run_mapping_audit，沿用显式repository/interpreter和
  参数列表，保留返回码、错误及输出路径语义。外部仓库VALIDATION_API.md记录公开契约。
- 新增`plane/tests/test_mujoco_process_api.py`检查literal参数传递、非零返回码和缺失入口。
  对保存的闭链轨迹实际运行旧/新MuJoCo审计，全部JSON字段一致，覆盖左右腿2个样本；
  证据`mapping_api_equivalence.json`及`mapping_public_api.json`位于架构验收outputs目录。
- 全套151项测试通过；训练仓库静态图231模块/1040边，循环与迁移层违规为0。
  此图尚不覆盖外部仓库完整内部依赖，不能声称两仓库总体验收完成。
- 更新根ARCHITECTURE与dependencies.json。未训练；tree probe的monkey patch仍待迁移。

## 第十五阶段：Tree probe移除monkey patch

- sim2sim仓库`sim2sim_parity.py`增加可选command_at/after_step回调，默认调用行为保持原样；
  新增公开`probe_tree_ramp.py`，接管原训练侧probe，使用time-only命令及显式步后测量。
- 训练侧`tools/probe_tree_ramp.py`成为薄CLI，显式传入training-root；
  `adapters/mujoco/api.py`新增run_tree_probe，消除跨仓库sys.path和全局函数替换。
- 扩展`test_mujoco_process_api.py`，更新两仓库接口/架构文档和dependencies.json。
- 相同ONNX/XML、forward=1、34000步实际对照：全部trace、metrics、结果字段精确一致，
  仅wall_seconds不同。证据为outputs/architecture_refactor_20260923下
  tree_before.json、tree_after.json、tree_api_equivalence.json。
- 训练侧152项测试通过，sim2sim既有2项测试通过；静态循环仍0。
  回调目前暴露原始model/data，只读约束是契约而非强制防护，已如实记录；外部runner内部
  拆分和完整两仓库审计仍待完成，goal未结束。未训练。

## 第十六阶段：MuJoCo共用模型测量

- 外部仓库新增`model_measurements.py`，从`lqr_deploy.py`迁出ModelRefs、模型名称解析、
  传感器及接触计数8个类/函数；旧模块保留显式兼容导出。
- `sim2sim_closed_policy.py`直接导入共用测量模块；依赖为policy/LQR→measurements→
  guide-wheel配置/MuJoCo，不再通过LQR控制器获取这些测量实现。
- 新增`tests/test_model_measurements.py`及`MODEL_MEASUREMENTS.md`说明公开入口、图与
  剩余初始化/保护耦合。没有移动资产、改变传感器顺序或碰撞规则。
- 对50份实际闭链轨迹状态执行原函数与新函数，ModelRefs、全部sensor、全部geom接触、
  轮支撑及非轮接触计数精确一致；证据在训练outputs架构目录的
  `model_measurements_equivalence.json`。
- 外部完整3项测试通过，包含新增独立导入/兼容测试。
  PhysicalLqr初始化和保护函数仍待拆分，尚不宣布完整解耦。

## 第十七阶段：MuJoCo物理保护所有权

- 外部仓库新增`physical_guards.py`，迁出5个状态保护函数和4个原阈值；
  修改`lqr_deploy.py`为兼容导出，`sim2sim_closed_policy.py`直接依赖保护模块。
  依赖方向policy/LQR→guards→measurements，保护不依赖控制器或施加任何补偿。
- AST核对全部函数签名/函数体与迁移前一致；记录在
  `plane/outputs/architecture_refactor_20260923/physical_guards_equivalence.json`。
- 新增外部`tests/test_physical_guards.py`覆盖腿长、腿差和闭链残差边界，完整外部测试
  10项通过；更新MODEL_MEASUREMENTS.md说明入口和原阈值。未训练。
- 初始化仍依赖PhysicalLqr，整体goal保持未完成。

## 第十八阶段：初始姿态投影

- 外部新增`stance_initialization.py`，从lqr_deploy迁出闭链投影及3个原配置常量；
  lqr_deploy保留兼容导出，sim2sim_closed_policy直接导入初始化入口。
- 模块显式接收model/data/refs/hip_targets，依赖measurements/guards/SciPy，不创建LQR。
  修改外部MODEL_MEASUREMENTS.md记录输入所有权和仍存在的构造器耦合。
- 原函数与新函数对真实MJCF、默认及显式目标各执行一次投影；qpos/qvel/ctrl/sensor/
  site_xpos/qfrc_constraint精确一致，证据为架构outputs下stance_equivalence.json。
- 新增外部tests/test_stance_initialization.py，验证实际投影和错误目标拒绝；外部11项
  测试通过。未训练；equilibrium torque求解与LQR线性化仍待分离，goal未完成。

## 第十九阶段：静态力平衡接口

- 外部新增`equilibrium_forces.py`，从PhysicalLqr迁出actuator力映射和静态平衡求解；
  修改lqr_deploy为显式委托，原外力方法作为回调传入，保留MuJoCo调用与clip顺序。
- 新增外部tests/equilibrium_reference.py冻结原方法，test_equilibrium_forces.py在
  真实MJCF上比较零外力/双侧active-link外力两种情形的结果及全部关键副作用张量。
- 外部13项测试通过；更新MODEL_MEASUREMENTS.md的公开入口和剩余初始化限制。
  未训练，policy对PhysicalLqr构造器的最终移除仍需后续验证。

## 第二十阶段：初始化副作用实证

- 新增外部tests/test_policy_initialization_equivalence.py，对实际MJCF分别运行
  完整PhysicalLqr初始化与独立静态求解，按runner时序恢复后，参考qpos/qvel/control及
  qacc、qacc_warmstart、约束力、传感器、time全部精确一致。
- 数值差记录保存在outputs/architecture_refactor_20260923/initialization_side_effect_audit.json，
  每项最大绝对差均为0；外部完整14项测试通过。
- 发现不能直接删除的行为边界：旧构造器执行Riccati可解性检查。为遵守严格等价，暂未
  删除该拒绝路径。更新外部MODEL_MEASUREMENTS.md，后续先独立线性化/校验职责。
  本阶段产出是可重复运行的副作用证据，不宣称policy初始化已完全解耦；未训练。

## 第二十一阶段：线性化与可解性检查

- 外部新增lqr_linearization.py，PhysicalLqr.linear_lqr改为显式传入模型、索引和权重。
  新模块不依赖控制器状态机，保留有限差分、Q/R矩阵构造、Riccati及最终gain求解顺序。
- 新增tests/linearization_reference.py冻结旧方法，tests/test_lqr_linearization.py
  核对真实模型gain/C矩阵逐元素精确一致，并注入Riccati失败验证异常文本/cause和
  Jacobian写入顺序；外部完整15项测试通过。
- 更新MODEL_MEASUREMENTS.md的公开接口与限制；未训练。初始化最终替换仍待完成，
  当前不删除PhysicalLqr调用、不跳过Riccati拒绝路径。

## 第二十二阶段：Policy初始化替换

- 外部新增policy_equilibrium.py，sim2sim_closed_policy.py改用PolicyEquilibrium，
  不再从lqr_deploy导入或构造PhysicalLqr。保留参考状态、速度测量、Riccati和原索引检查。
- 新增controlled_state_indices.py及linearization_config.py，lqr_deploy改为兼容导入
  共同的索引校验和权重，避免policy依赖控制器状态机；没有改变默认XML路径或物理参数。
- 新增tests/test_policy_equilibrium.py验证参考状态、gain、C矩阵、索引与原构造器
  精确一致，并验证独立导入不会加载lqr_deploy；完整外部17项测试通过。
- 同一10200 ONNX、tree_zero、1m/s、2000步闭链仿真前后全部trace/metrics/结果字段
  精确一致，仅wall_seconds不同，确实完成2000步。证据为架构outputs中的
  policy_init_before.json、policy_init_after.json、policy_initialization_equivalence.json。
- 更新外部MODEL_MEASUREMENTS.md当前入口，明确剩余重复组装和完整审计任务；未训练。
  本阶段完成policy构造器解耦，不代表整体goal完成。

## 第二十三阶段：外部依赖审计补齐

- 外部新增tools/check_architecture.py、dependencies.json、ARCHITECTURE.md及
  tests/test_architecture.py；静态扫描根模块/tools/tests，含函数内导入和包初始化。
- 新核心模块使用显式本地依赖白名单，禁止measurements/guards等反向依赖LQR；
  测试用构造的函数内循环和非法控制器依赖验证检查器确实会报告错误。
- 当前外部35模块、55依赖边、0循环、0受管模块越界；完整外部19项测试通过。
- 外部图明确标注legacy未分层、私有诊断调用、可写回调和重复组装等剩余工作；
  训练根ARCHITECTURE链接到外部审计职责，避免将单仓库结果推广到两仓库。
- 本阶段未训练，完整goal仍有环境/工作流/公共接口整理及最终验证待完成。

## 第二十四阶段：公开策略测量接口

- 外部新增policy_measurements.py，从sim2sim_closed_policy迁出7项只读测量函数；
  runner保留旧别名，validate_policy改为直接引用公开模块，不再访问runner私有测量。
- 更新外部tools/check_architecture.py允许依赖、dependencies.json、架构和测量指南。
- 实际CLI首次执行暴露模块别名与统计字典同名，已修正；重新运行2000步闭链验证，
  除耗时和源码hash外全部JSON字段精确一致。证据为public_measurements_fixed.json和
  public_measurements_equivalence.json，位于架构outputs目录。
- 外部19项测试通过，依赖图36模块/59边，循环和受管模块越界为0。未训练；
  只读回调快照、重复索引组装和训练侧剩余分层仍待完成。

## 第二十五阶段：共用索引组装与heading测量

- 外部controlled_state_indices.py新增ControlIndices与build_control_indices，将两份
  初始化索引组装归一；lqr_deploy.py/policy_equilibrium.py保持逐字段赋值，未使用mixin
  或动态属性注入。model_measurements.py统一forward_direction/forward_speed。
- 首次回归发现局部变量遮蔽lqr_state_indices函数，修正后外部19项测试全部通过。
- 再运行2000步闭链轨迹，与迁移前除wall_seconds外所有字段精确一致；证据为架构
  outputs下shared_indices_after.json和shared_indices_equivalence.json。
- 更新外部架构/模块文档和dependencies.json：36模块/61边，循环和受管违规均0。
  未训练；回调快照及训练侧环境/工作流等剩余目标仍待完成。

## 第二十六阶段：Motion评估工作流

- 新增workflows/motion_evaluation.py，将原app闭包中的评估执行、来源校验和结果汇总
  移入显式入口。EvaluationOptions为冻结配置，stage命令、文件/进程/digest能力由app注入。
- 修改app/motion_supervisor.py为薄组装；adapters/artifacts/job_files.py补齐JSON接口。
  workflow只依赖evaluation，核心层不导入app/adapters，也不读取环境变量。
- 新增test_motion_evaluation_workflow.py验证缓存/缺失结果分支、SHA拒绝和geometry保留。
  用真实历史rollout对照旧闭包，稳态与transition的返回值及写入JSON字节全部一致，
  证据motion_workflow_equivalence.json位于架构outputs目录。
- 全套155项测试通过，图232模块/1043边，循环/迁移层违规均0；更新模块指南和依赖图。
  未训练；轮次选择、export和其余环境/工作流仍待继续迁移。

## 第二十七阶段：Isaac基础环境归属

- 将envs/base/base_task.py和utils/terrain.py实现原样迁至
  adapters/isaacgym/base_task.py、terrain_generation.py；旧模块显式兼容导出。
  legged_robot直接依赖适配器，目录表达Gym/viewer/地形工具的所有权。
- 更新地形模块指南及根ARCHITECTURE，明确保留的Torch JIT进程级开关，而不误报全局状态清零。
- 全套155项测试通过；静态图234模块/1053边，循环和受管层违规均为0。
- 真实64env/seed11/1iteration smoke完成3072步；与阶段二checkpoint比较，模型和
  两组optimizer及iter精确相等，源码快照hash验证通过。证据lifecycle_smoke.log与
  lifecycle_smoke_equivalence.json位于架构outputs目录。这是验证smoke，没有恢复长训练。
- actor创建、buffer管理及整体生命周期继续迁移，整个goal尚未完成。

## 第二十八阶段：Robot资产加载边界

- 新增adapters/isaacgym/robot_asset.py，AssetOptions、加载及metadata查询从
  legged_robot._create_envs迁出；环境显式接收LoadedRobotAsset字段，actor循环不变。
- 新增test_robot_asset.py验证全部资产选项、Gym查询顺序和原始属性对象身份，
  保留body_count随后按名称数覆盖的历史语义。全套156项测试通过。
- 更新terrain_creation模块指南和dependencies.json：235模块/1057边，静态循环及
  受管层违规为0；未训练。actor随机化/创建循环仍由环境持有，后续继续解耦。

## 第二十九阶段：Actor创建循环

- 新增adapters/isaacgym/actor_creation.py，legged_robot显式传入句柄、origin、属性对象、
  三个处理回调及结果列表；实例循环保持原位置扰动与shape/DOF/body操作顺序。
- 环境仍拥有随机化buffer，适配器只拥有Gym调用顺序；不传入整个环境对象、不用mixin。
  更新terrain_creation模块指南和dependencies.json，图236模块/1061边，无循环/受管违规。
- 全套156项测试通过。真实64env/seed11/1iteration smoke后模型、主/extra optimizer、
  iter与阶段二基线精确一致，184份源码快照hash校验通过；证据actor_smoke.log和
  actor_smoke_equivalence.json在架构outputs目录。未恢复长训练。
- buffer初始化、随机化策略与整体目标的剩余工作仍待继续处理，goal未完成。

## 第三十阶段：Gym共享状态张量

- 新增adapters/isaacgym/state_tensors.py，以SimulationTensors显式返回8个状态字段；
  legged_robot._init_buffers只负责逐字段接收，获取/刷新/wrap/view顺序不变。
- 新增test_state_tensors.py验证Gym调用顺序、DOF/root/contact/body共享写入及dof_acc
  独立存储；全套157项测试通过。未将仿真视图复制为独立buffer。
- 新增docs/modules/simulation_tensors.md，更新根架构导航和依赖JSON：237模块/1065边，
  循环和受管层违规均0。未训练；历史/随机化buffer和整体剩余职责继续迁移。

## 第三十一阶段：显式命令指标存储

- 新增domain/commands/metrics_state.py，将25个累计buffer定义为具名CommandMetrics，
  legged_robot._init_buffers逐字段赋值，去除这处动态setattr。
- 保留逐字段分配顺序和reset列表引用相同张量的语义，不共享不同任务存储；
  新增test_command_metrics_state.py验证25字段、别名、实例隔离和RNG不变。
- 全套158项测试通过；依赖图238模块/1069边，循环和受管层违规均0。
  更新共享张量模块指南和依赖JSON，未训练；累计公式/其他buffer等剩余工作继续推进。

## 第三十二阶段：命令指标累计

- 新增domain/commands/tracking_metrics.py，legged_robot仅传入具名CommandMetrics及
  显式状态/阈值/计算回调；按原顺序原地更新，不改method与legacy的slip定义。
- 新增test_tracking_metrics_equivalence.py及冻结fixture，覆盖两种模式三次累计，
  全部25字段及zero/wheel回调顺序与旧方法一致。
- 全套160项测试通过，图239模块/1073边，循环和迁移层违规为0。更新张量指南与依赖图，
  未训练；整体goal仍有环境/工作流/完整文档验收待完成。

## 第三十三阶段：评估公开API命名

- evaluation_setup公开build_evaluation_args/disable_evaluation_randomization，policy_io
  公开set_fixed_command_ranges；experiments/primitives公开apply_method_randomization。
  旧下划线导出在原适配器/脚本/配置包装中保留兼容别名，函数体不变。
- 更新4个评估CLI（evaluate_standing、evaluate_policy_comparison、isaac_command_grid、
  isaac_parity_trace）、3个audit工具（height_rollout、height_update、ppo_update）、
  3个recipe（low_speed、stand_balance、fudan_stand）及policy_experiments包装的调用。
- 全套160项测试通过，已无这些旧私有名字的跨模块import；静态循环和受管违规均0。
  更新配置指南和dependencies.json；未训练，其他未完成职责保持原计划继续处理。

## 第三十四阶段：消除被覆盖的方法定义

- legged_robot.py原有两个_process_dof_props定义，前者只有历史注释，Python实际只使用
  后者。删除前一不可执行定义，AST确认生效方法的签名和函数体完全未变。
- plane/tests/test_architecture.py新增全包重复类方法检查，显式排除同名property
  setter/deleter；避免新窗口读到实际上不会执行的方法而误判物理配置。
- 全套161项测试通过，更新dependencies.json源码行号；循环/受管违规为0。
  未训练，未修改armature/DOF限位语义，剩余架构goal继续进行。

## 第三十五阶段：DOF物理属性入口

- 新增adapters/isaacgym/dof_properties.py，将限位读取/soft limit计算与armature
  名称匹配分成两个公开函数；legged_robot._process_dof_props仅协调env0初始化和应用。
- 新增test_dof_properties.py覆盖限位数值、原属性不变、armature原地写入、首匹配优先、
  未匹配清零及既有打印。全套162项测试通过，静态图240模块/1077边，无循环/受管违规。
- 更新地形/资产模块指南与dependencies.json，未训练；刚体/shape随机化等仍待迁移。

## 第三十六阶段：刚体随机化

- 新增adapters/isaacgym/body_properties.py，显式BodyRandomizationState和配置输入；
  legged_robot回调只组装输入并接收引用，保留mass/COM/inertia顺序及env0批量采样。
- 新增test_body_properties_equivalence.py和冻结fixture，8种开关组合、每种3个环境的
  属性、buffer和Torch/NumPy RNG状态与原实现一致。全套170项测试通过。
- 更新资产模块指南，记录base_mass先于inertia缩放的旧语义，未顺手改变诊断口径。
  图241模块/1081边，无循环/受管违规；未训练，shape随机化等剩余职责继续处理。

## 第三十七阶段：Shape随机化及联合回归

- 新增adapters/isaacgym/shape_properties.py，legged_robot显式传入配置与系数state；
  保留64bucket、CPU bucket_ids、设备随机数及属性写入顺序。
- 新增test_shape_properties_equivalence.py和冻结fixture，四种开关组合、五个环境的
  属性/系数/RNG均与原方法一致；全套174项测试通过。
- 真实64env/seed11/1iteration联合smoke，覆盖近期DOF/body/shape/指标拆分；模型、
  两组optimizer、iter与阶段二基线精确一致，190份源码快照hash全部通过。
  证据为架构outputs中的randomization_smoke.log、randomization_smoke_equivalence.json。
- 更新资产模块指南与依赖JSON：242模块/1085边，无循环/受管违规；未恢复长训练。
  整体goal剩余工作继续进行。

## 第三十八阶段：终止判定边界

- 新增domain/termination.py，显式TerminationState与状态张量输入；legged_robot仅组装
  输入/接收输出，reset_idx仍拥有重置执行。未改warmup、grace、timeout或edge语义。
- 新增test_termination_equivalence.py和冻结fixture，8种模式组合各连续5次执行，
  新环境方法与旧方法全部状态张量精确一致；全套182项测试通过。
- 更新根架构入口、新增docs/modules/termination.md及依赖JSON：243模块/1088边，
  静态循环和受管违规均0；未训练，其他生命周期及整体目标继续推进。

## 第三十九阶段：物理状态reset

- 新增adapters/isaacgym/state_reset.py，DOF/root indexed写回由适配器负责，
  legged_robot保留原方法委托；不搬走reset_idx的课程/episode协调逻辑。
- 新增test_state_reset.py验证custom/普通origin分支、RNG、未选中行不变、共享张量及
  int32索引写回；全套184项测试通过。纠正旧DOF reset随机缩放注释，代码仍用原默认位置。
- 更新张量模块指南和依赖图：244模块/1092边，无循环/受管违规；未训练，goal继续。

## 第四十阶段：课程窗口累计

- 新增domain/commands/curriculum_window.py，负责标量窗口创建与累计；legged_robot
  保留启用判定、有效episode筛选、episode计数和调用顺序，显式传入CommandMetrics。
- 新增test_curriculum_window.py验证有效索引、reverse绝对值、空集合、窗口独立性及
  源buffer不变。全套185项测试通过；图245模块/1096边，循环/受管违规均0。
- 更新课程模块指南和dependencies.json；未训练，整体goal仍按原范围继续推进。

## 第四十一阶段：ONNX等价与完整验收缺口

- 用当前learning实现执行verify_onnx.py，对原10200 checkpoint/历史ONNX验证256样本，
  max_abs_error=7.62939453e-06、mean_abs_error=6.95020105e-07，原容差下通过。
- 当前export_onnx.py重新导出同一checkpoint到架构outputs，未覆盖原成果。新旧文件
  SHA256完全相同；batch=1/8/256输出逐元素相同，接口仍obs25/history125/actions6。
  证据onnx_original_verify.log、onnx_reexport.log、onnx_reexport_equivalence.json。
- 新增docs/architecture/completion_audit.md按完整原目标列出证据与剩余缺口，根
  ARCHITECTURE链接该表，明确仍有环境/监督器、回调快照、文档和版本记录工作。
- 本阶段未修改训练行为、未训练；ONNX验证不等于运动能力验收或goal完成。

## 第四十二阶段：闭链只读测量快照

- 外部新增measurement_snapshot.py冻结数值快照，sim2sim_closed_policy新增on_measurement；
  validate_policy改用该接口，不再在步后回调持有实时model/data。初始化回调仍可按契约写状态。
- 新增test_measurement_snapshot.py验证字段冻结、嵌套tuple不可写及与live状态隔离。
  完整外部20项测试通过；图38模块/68边，无循环/受管违规。
- 2000步closed对照除耗时和源码hash外全部JSON字段精确一致，证据snapshot_after.json、
  snapshot_equivalence.json在架构outputs目录。
- 更新外部公开接口/架构、两仓库验收台账。旧after_step保留兼容并明确顺序，tree快照
  尚待迁移；未训练，不因此宣称完整goal已完成。

## 第四十三阶段：Tree只读测量快照

- 外部新增tree_measurement_snapshot.py，sim2sim_parity支持on_measurement，
  probe_tree_ramp改用冻结值快照，不再持有运行中的model/data。
- 扩展test_measurement_snapshot.py检查速度坐标、字段冻结和底层数组/contacts隔离；
  外部21项测试通过。34000步对照除耗时外全部结果精确一致，证据tree_snapshot_after.json、
  tree_snapshot_equivalence.json位于架构outputs目录。
- 更新外部公开接口/架构、依赖图和验收台账；39模块/70边，循环/受管违规0。
  旧after_step仅保留兼容，仓库内两个probe均用快照。未训练，整体goal继续。

## 第四十四阶段：跨进程依赖清单与防回退检查

- 新增docs/architecture/process_interfaces.json，记录三个训练入口→adapter→外部CLI、
  输入/输出协议、解释器/仓库定位以及旧after_step的兼容原因、边界和移除条件。
- 新增test_cross_repository_boundary.py检查包装实际导入指定adapter，并扫描训练包/tools
  禁止重新导入独立MuJoCo内部模块；外部test_architecture.py检查自有probe只用快照回调。
- 更新根ARCHITECTURE和完整验收台账，区别静态Python依赖图与跨进程协议。
  训练侧187项、外部22项测试通过；未训练，剩余分层和文档/版本任务继续推进。

## 第四十五阶段：导出工作流与显式ports

- 新增workflows/policy_export.py，app/motion_supervisor的export只组装路径与执行器；
  导出→batch256数值校验的命令、顺序、失败传播保持原样。
- 新增ports/processes.py和ports/artifacts.py，声明PythonJob/EvaluationArtifacts，
  motion_evaluation改用接口类型，不依赖具体adapter。
- 新增test_policy_export_workflow.py覆盖命令参数及两阶段失败；全套190项测试通过。
  补充robot解释器下仅运行评估/导出工作流6项测试通过，证明无需导入Isaac Gym。
- 更新监督器指南及依赖JSON：248模块/1108边，无循环/受管违规；未训练，轮次工作流
  和其余验收缺口继续处理。

## 第四十六阶段：根入口与历史分离

- 重写根README为当前架构导航、目录职责、固定接口及检查命令；根AGENTS保留稳定的
  项目/物理/环境/日志/对称性/验收/修改规则，增加本轮明确约束，不再优先指向旧训练快照。
- 整理前两份文件全文原样保存在docs/history/README_before_architecture_20260923.md、
  AGENTS_before_architecture_20260923.md，包含原有dirty内容；新增history/README解释
  快照日期和原相对路径基准。历史记录未删除，不再冒充当前状态。
- 新根README/AGENTS全部本地链接现场检查存在；git diff --check通过。更新验收台账。
  本次仅文档，不重复运行训练测试；goal仍有未完成代码/最终验证/版本记录工作。

## 第四十七阶段：实验定义去除可变全局表

- 新增experiments/definitions.py，3张实验定义表直接构造成递归只读映射；
  primitives导入静态定义，并为legacy manifest复制独立嵌套字典，数值/tuple类型不变。
- 新增test_experiment_definitions.py验证嵌套只读、manifest仍可JSON序列化/修改，且
  修改某次optimizer不再污染下次实验。此隔离行为落实去除隐式全局状态的要求。
- 全套192项测试通过；依赖图249模块/1111边，无循环/受管违规。更新实验模块指南，
  未改变配置默认数值、未训练，其他全局兼容边界仍待最终审查。

## 第五十三阶段：删除冗余可写测量接口（2026-09-24）

- 按用户最新指示，兼容旧用法不再作为硬要求；AGENTS、兼容指南和验收台账落地该约束。
- 扫描确认自有调用方全部使用快照，移除外部sim2sim_closed_policy.py、sim2sim_parity.py
  的after_step参数与分支；on_measurement为唯一测量通道，不更改物理执行路径。
- 扩展外部tests/test_architecture.py禁止runner再次暴露旧参数；完整外部23项测试通过。
  更新外部VALIDATION_API/ARCHITECTURE、dependencies.json及训练侧process_interfaces.json。
- 未训练；后续继续清理实际无调用的兼容包装，保留模型与行为等价约束，goal未完成。

## 第四十八阶段：旧学习包导入兼容

- 补回rsl_rl/modules/__init__.py遗漏的4项显式构造器导出及原2项__all__，全部指向
  learning实现，不复制类或恢复注册副作用。
- 新增test_legacy_learning_imports.py验证网络包级/叶子/API、PPO、runner、storage的
  旧新对象身份相同；全套194项测试通过，图249模块/1115边，无循环/受管违规。
- 新增docs/modules/compatibility.md，明确保留入口及全局task_registry/自动注册的移除，
  不以CLI兼容冒充任意外部Python导入兼容。根架构链接该指南；未训练，goal继续。

## 第四十九阶段：候选两阶段筛选

- 新增workflows/candidate_screening.py，app/motion_supervisor委托5个checkpoint的
  单seed筛查及top2三seed验收，保持safe/posture门槛和tuple路径平分顺序。
- 新增test_candidate_screening.py覆盖调用顺序、第二轮拒绝、无安全候选及失败传播；
  全套197项测试通过，图250模块/1118边，无循环/受管违规，git diff --check通过。
- 更新监督器模块指南。未训练；动态验收后的接受/回退和source更新仍待整理。

## 第六十五阶段：MuJoCo配置读取归启动层（2026-09-24）

- 外部model_measurements.py移除导入时读取的全局guide contract与名称常量，build_refs
  和optional_guide_wheel_refs显式接收契约。lqr_deploy、sim2sim_closed_policy、
  sim2sim_closed_chain三个运行入口加载并注入；7个相关测试调用同步迁移。
- 外部23项既有测试通过，新增导入隔离测试后test_model_measurements两项通过，证明
  导入测量模块不加载guide_wheel_mjcf。无需保留旧隐式API。
- 更新外部模型指南、架构图、依赖规则/JSON；39模块/78边，无循环/受管违规。
  未训练，剩余验收工作继续推进。

## 第六十四阶段：Torch进程副作用显式化（2026-09-24）

- 新增app/torch_runtime.py，TaskRegistry.make_env显式设置两个历史JIT开关；
  adapters/isaacgym/base_task.py删除构造器内的隐式进程设置，不引入反向依赖。
- 全套246项测试通过；64env/seed11/1iteration smoke模型、两组optimizer与阶段二
  基线精确一致，源码快照hash通过。证据torch_runtime_smoke.log及对应equivalence.json。
- 更新配置/地形指南，明确直接构造环境的新要求及旧Gym兼容边界；依赖审计通过。
  未恢复长训练，剩余全局配置及最终架构验收继续推进。

## 第六十三阶段：显式导入与删包装联合回归（2026-09-24）

- legged_robot.py将Isaac torch_utils通配符改为6个实际使用的显式函数；play_export.py
  删除未使用的通配符导入。test_architecture.py增加全训练包禁止star import检查。
- 全套246项测试通过。实际64env/seed11/1iteration smoke验证近期转发包删除、数学迁移、
  高度采样及入口路径，模型/两组optimizer/iteration与阶段二基线精确一致。
  证据public_imports_smoke.log和public_imports_smoke_equivalence.json位于架构outputs。
- dependencies.json更新，204模块/891边，无循环/受管违规/缺失本地模块；未恢复长训练，
  完整goal剩余内容继续按台账推进。

## 第六十二阶段：高度采样计算（2026-09-24）

- 新增domain/geometry/height_sampling.py，从legged_robot提取高度采样算术；
  输入显式列出网格、位姿、采样点和配置，不传整个环境对象。
- 新增test_height_sampling_equivalence.py与冻结fixture，覆盖4种mesh和3种env_ids，
  张量结果及原有NameError/多元素Tensor错误精确一致，未顺手修正子集reshape语义。
- 全套245项测试通过；图204模块/891边，无循环/受管违规/缺失模块。更新地形指南和
  dependencies.json；未训练，goal仍按完整验收台账推进。

## 第六十一阶段：环境数学依赖下沉（2026-09-24）

- 删除utils/math.py；quat_apply_yaw迁至domain/geometry/rotations.py，直接复刻本机
  Isaac normalize/quat_apply运算顺序（含JIT）；wrap_to_pi复用已有domain实现。
  无调用的torch_rand_sqrt_float删除，legged_robot移除其导入。
- 新增test_yaw_rotation_equivalence.py对照实际Isaac函数，覆盖两种形状/零四元数与
  输入不变；全套233项测试通过。check_architecture禁止envs→utils。
- 图203模块/883边，无循环/受管违规/缺失模块；更新架构/兼容指南与验收台账，未训练。

## 第六十阶段：环境与组装层依赖约束（2026-09-24）

- 核对依赖JSON后，tools/check_architecture.py将envs/utils/app纳入允许方向规则：
  禁止envs→app/experiments/learning、utils→envs、app→scripts，明确环境协调器可用adapter。
- 扩展test_architecture.py检查这些禁止边和合法adapter边，5项架构测试通过；
  当前204模块/882边，无循环、层级违规、核心仿真器导入或缺失模块。
- 更新ARCHITECTURE、验收台账和依赖JSON；仅规则/文档变更，未重复训练测试或启动训练。
  tools/scripts残余入口和envs→utils.math仍需继续整理，整体goal未完成。

## 第五十九阶段：缺失模块依赖检测（2026-09-24）

- tools/check_architecture.py增加missing_local_imports并纳入退出码，解析绝对及函数内
  相对导入，允许有效namespace package，防止删包装后缺失模块被原图静默忽略。
- plane/tests/test_architecture.py新增合成缺失导入与合法namespace/属性导入用例，
  全套231项测试通过；当前204模块/882边，无循环/受管违规/缺失本地模块。
- 更新ARCHITECTURE和dependencies.json，明确属性导入/动态加载仍需运行验证，未夸大
  静态检查覆盖范围；未训练，剩余架构goal继续。

## 第五十八阶段：删除环境基础转发（2026-09-24）

- 删除envs/base下command_sampling、command_curriculum、height_commands、reward_terms、
  start_stop_commands、base_task六个转发文件。legged_robot内4处函数级相对导入改为
  domain.commands明确入口，避免高度/启停/fixed-bank分支只在运行时发现缺失模块。
- test_architecture.py扩展移除路径检查并解析相对导入；全套230项测试通过，新增检查
  后4项架构测试再次通过。图204模块/882边，无循环/受管违规。
- 更新兼容/地形模块指南和依赖JSON，未训练。后续仍需收敛未完成分层及最终验收。

## 第五十六阶段：配置唯一入口（2026-09-24）

- 删除envs/base/base_config.py、legged_robot_config.py及envs/wheel_legged/
  wheel_legged_config.py三处转发；legged_robot直接引用contracts配置。
- app/task_registry.py保存源码改为contracts实际实现，新增test_registry_source_paths.py
  实际执行save_cfgs，核对两个配置与terrain_generation的输出字节一致。
- 既有全套229项测试通过，新增源码保存测试在robot环境单独通过；图225模块/1002边，
  无循环/受管违规。更新架构/兼容说明及依赖图，未训练，goal继续。

## 第五十五阶段：删除utility转发层（2026-09-24）

- 删除utils/helpers.py、task_registry.py、terrain.py三个转发文件；6个测试文件
  （dynamic_equivariance、legacy_speed2、height_course、motion_geometry、legacy_anchors、
  motion_switch_recipe）直接引用contracts.config_serialization。
- app/task_registry.py的源码快照保存路径改为实际terrain_generation.py，新增架构测试
  禁止生产/测试代码再导入这3个旧模块；同步配置/地形/兼容指南及根架构说明。
- 全套229项测试通过；图228模块/1011边，无循环/受管违规；未训练，goal继续。

## 第五十四阶段：删除旧学习命名空间（2026-09-24）

- 扫描确认生产代码无旧学习导入后，删除plane/wheel_legged_gym/rsl_rl下21个纯转发
  Python文件；实际学习实现全部保留在learning，原改动可从备份/Git追溯。
- 删除test_legacy_learning_imports.py，新增test_learning_entrypoints.py验证唯一入口
  并禁止重新导入旧包；compare_source_contract.py显式区分上游旧路径和本地新路径，
  同时修正其配置对照指向contracts实际实现，不比较包装壳。
- 全套228项测试通过；实际10200 checkpoint/历史ONNX、batch256数值核验通过，
  最大误差7.62939453e-06，与迁移前相同。证据learning_namespace_verify.log。
- 图从252模块1124边减少为231模块1049边，无循环/受管违规；更新README、ARCHITECTURE、
  兼容指南和dependencies.json。未改模型文件、未训练，goal继续。

## 第五十阶段：动态候选验证（2026-09-24）

- 中断恢复后检查进程，未发现遗留pytest或训练进程；继续已落地的候选筛选工作。
- 新增workflows/candidate_validation.py，app/motion_supervisor显式委托附加动态门槛，
  保留启停→switches顺序、原result更新时机及AND语义，不更新source或导出模型。
- 新增test_candidate_validation.py覆盖16种静态/动态/启用/阶段组合和异常传播；
  全套214项测试通过（1项既有warning）。依赖图251模块/1121边，无循环/受管违规。
- 更新监督器指南和dependencies.json；命令为fudan_leg Python执行pytest plane/tests，
  python3 tools/check_architecture.py --write docs/architecture/dependencies.json。
  未训练；接受/回退及最终架构验收仍未完成，goal保持原完整范围。

## 第五十二阶段：配置实例隔离（2026-09-24）

- 现场确认goal已恢复active。contracts/base_config.py在嵌套配置实例化时深复制容器，
  去除list/dict默认值跨实验共享；保留原数值、字段和构造入口。
- 新增test_config_instance_isolation.py，验证嵌套list/dict/tuple与robot/PPO配置的实例
  修改不会污染类默认值或后续实例。全套228项测试通过，依赖检查无循环/受管违规。
- 64env/seed11/1iteration真实smoke：模型、两组optimizer、iteration与阶段二基线
  精确一致，200份源码快照hash通过。证据config_isolation_smoke.log和
  config_isolation_smoke_equivalence.json位于架构outputs目录；未恢复长训练。
- 更新configuration模块指南及完整验收台账，整体goal继续保持原范围。

## 第五十一阶段：候选决策值与focus（2026-09-24）

- 用户明确要求恢复执行后核对goal，工具仍返回blocked；未谎称状态已active，继续在
  用户授权内推进代码。自动连续执行的Resume状态由界面控制。
- 新增workflows/candidate_decision.py及test_candidate_decision.py；修改
  app/motion_supervisor.py显式使用决策结果，保持严格0.01改善门槛、接受与改善的区别、
  focus顺序/重复上限及导出成功后才更新accepted的时机。
- 全套226项测试通过；python3 tools/check_architecture.py --write
  docs/architecture/dependencies.json通过，252模块/1124边，无循环/受管违规。
- 更新监督器指南与依赖图；未训练。整体目标尚未完成，仍需关闭完整验收台账缺口。

## 第五十七阶段：删除实验转发目录（2026-09-24）

- 删除envs/wheel_legged下15个纯实验转发文件；调用方逐符号直达实际app/experiments/
  artifacts入口，不引入统一万能facade。配置/配方/优化器行为未改。
- 修改调用方完整清单：
  - `plane/wheel_legged_gym/scripts/evaluate_standing.py`
  - `plane/wheel_legged_gym/scripts/train.py`
  - `plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py`
  - `plane/tests/test_legacy_yaw.py`
  - `plane/tests/test_stop_retention.py`
  - `plane/tests/test_speed2_stop.py`
  - `plane/tests/test_fudan_stand.py`
  - `plane/tests/test_encoder_anchor.py`
  - `plane/tests/test_clean_observation.py`
  - `plane/tests/test_h3_speed1.py`
  - `plane/tests/test_legacy_speed2.py`
  - `plane/tests/test_bilateral_geometry.py`
  - `plane/tests/test_command_diagnostics.py`
  - `plane/tests/test_stand_balance.py`
  - `plane/tests/test_legacy_urdf.py`
  - `plane/tests/test_policy_experiments.py`
  - `plane/tests/test_legacy_anchors.py`
  - `plane/tests/test_low_speed.py`
  - `plane/tests/test_h3_low_speed.py`
  - `tools/audit_height_rollout.py`
  - `tools/audit_ppo_update.py`
  - `tools/audit_height_update.py`
- 全套230项测试通过，图210模块/902边，无循环/受管违规；更新README、ARCHITECTURE、
  兼容指南及依赖JSON。唯一残留旧路径为compare_source_contract的只读上游路径。
- 未训练；整体goal仍按验收台账推进。
阶段98后续补充：`continue_height_course.py`已迁移为app.continue_height_course.main(root)，
保留round/seed、screen-job等待、部分候选、三轮停滞、STOP和子job status协议；robot环境
入口可导入且未启动续训。新增test_continue_height_entrypoint.py验证CLI无Popen/导入无mkdir。
剩余直接子进程诊断仅audit_observation_noise，另有三个只读/job文件工具。

阶段98再次补充：`audit_observation_noise.py`已迁移为app.audit_observation_noise.main(root, argv)，
保留sampled/sampled_noisy模式和字面参数传递；新增3项边界测试，未运行实际诊断。
剩余tools仅3个旧job文件读写工具，不再有直接子进程诊断入口。

阶段98最终补充：wait_for_completion、summarize_policy_comparison、export_model_registry
已迁移到app同名入口，tools只保留CLI引导；新增job_tool_entrypoints测试验证导入不读写文件、
 不启动进程。旧文件格式的真实读写对照仍待最终验收，不运行历史任务。

补充：迁移后将test_height_continuation和test_policy_comparison_gate的旧tools路径导入
改为直接责任模块；没有保留已删除tools的Python导出别名。训练全套测试目前664 passed、
2 warnings；sim2sim此前24 passed。job文件真实历史样本未被改写或加载执行。

阶段99收口：tools全量AST扫描已无subprocess调用；所有工具只做CLI引导或只读入口，
进程/状态所有权归app/adapters。训练全套664 passed、2 warnings；依赖图249模块/1045边，
cycles、层级违规、仿真导入、缺失本地模块均为空。剩余仅是历史job文件真实样本读写验收、
最终smoke/已有等价证据汇总与最终提交，不再有未迁移的tools进程入口。

## 2026-09-25 阶段100：最终代表性验证与目标收口

- 最终训练启动 smoke：64 env、seed11、max_iterations=1、run_name=`architecture_final_smoke_20260925`，
  退出0并生成model_1.pt；日志位于`plane/outputs/architecture_refactor_20260923/final_smoke.log`。
- 最终测试：训练仓库664 passed、2 warnings；sim2sim仓库24 passed。依赖审计训练249模块/1045边、
  sim2sim39模块/78边，cycles、受管层级违规、核心仿真导入和缺失本地模块均为空。tools AST无
  subprocess调用，历史入口均为app薄CLI。
- ONNX、closed/tree固定步长等价、正常/保护拒绝双路径、原dirty Git归档和分阶段提交均已在本记录
  与验收台账索引；25D/125D/6D契约未改。
- 剩余外部边界是第三方Isaac/PyTorch运行时开关、pynput/GUI交互、历史job文件格式和HAPI通知，
  已由AGENTS、模块指南、process_interfaces和测试记录；不属于自有模块循环或向上依赖。
  未运行历史长任务，不以GUI/均值/reward宣称运动能力。

## 2026-09-25 Codex 入口与 review 收口

- 现场分支均为 `refactor/codex-readable-20260923_100634`。训练 HEAD `59396eb`，sim2sim HEAD
  `ee84200`；原有 README dirty changes、未跟踪快速入口和 `.deep-copilot/` 均保留。
- 完成两个 `CODEX_QUICKSTART.md`：目录职责、最小阅读、准确测试/审计/1 iteration smoke/
  34000步 MuJoCo 命令、启动副作用、保护目录、故障日志及25D/125D/6D契约。
- 新增 `process_review.md`，按25个 app 入口列出 import/main、job、lock、原子 status、STOP、
  child_pid、wait、恢复、hash、覆盖和通知；更新 supervisor_inventory 为当前入口索引。
- 重写 `completion_audit.md` 的12项状态、证据、验证命令、剩余边界与 Goal 影响；
  同步训练 ARCHITECTURE/README 和 sim2sim ARCHITECTURE/MODEL_MEASUREMENTS/README 的现状描述。
- 本轮未改 Python 源码，依赖图无变化，故 `dependencies.json` 保持原内容。
  `python3 tools/check_architecture.py`：249模块/1045边，cycles、layer、simulator import、
  missing local imports 全空；robot Python 审计 sim2sim：39模块/78边，cycles/layer 全空。
- 完整测试：fudan_leg Python `-m pytest plane/tests -q --disable-warnings --maxfail=1`：
  664 passed、2 warnings；robot Python `-m pytest tests -q`：24 passed。
  沿用阶段100的 final_smoke.log、final_onnx_equivalence.json 和
  closed_success_rejection_equivalence.json；文档修改无需重启 Isaac 或覆盖旧结果。
- 架构等价目标的自有模块项已关闭；第三方运行时、历史 job 文件格式、HAPI 通知
  仍是明确外部边界。新策略运动能力 gate 需要独立授权和完整验收，不能由架构测试推出。

## 2026-09-25 定向缺陷复核与纠偏

- 上一节“架构等价目标的自有模块项已关闭”结论过强：源码复核发现实际 CLI/root
  缺陷，故以 `completion_audit.md` 当前“明确边界/未完成”为准，不把这些缺口归为第三方。
- `app.compare_policy_versions.main(root)` 现在从显式 root 组装 PLANE/EVALUATOR、
  checkpoint、asset、cwd。`runner_sha256` 保持校验 app 源码；迁移前 manifest hash
  不匹配时拒绝恢复，不改旧 job。临时目录加 fake Popen 测试验证新 job 路径/hash，
  另用旧 hash manifest 验证拒绝恢复。
- `tools/export_model_registry.py` 现在传入 ROOT；`runpy` 加 fake main 验证 CLI 绑定，
  未触碰固定 `docs/data`。`app.audit_observation_noise` 不再改变子进程 cwd；
  不同调用目录的 fake subprocess 测试验证相对参数仍按调用者 cwd 解释。
- 三处生产代码依赖方向未变，依赖 JSON 无需重写。定向 11 项测试通过；
  完整训练测试 667 passed、2 warnings，sim2sim 测试 24 passed。
  训练依赖审计 249 模块/1045 边，sim2sim 39 模块/78 边，均无循环或受管层级违规。
  `git diff --check` 两仓库均通过。
  旧 manifest、logs、outputs、资产和 `.deep-copilot/` 均未清理或覆盖。
- 后续最小验收是历史 job 格式/恢复拒绝的隔离读写、其余重要 CLI/root 绑定、
  GUI 交互受控验证；未在本阶段自动启动长训练。
# 2026-09-25 fixed-height turn-envelope experiment (behavior change)

This is a separately authorized training experiment, not behavior-equivalent
refactoring. `TURN_ENVELOPE` adds a pure recipe, exact R10200 source validation,
an open-loop 25% turn command cohort, and an optional command-derived roll
reference. A retains the original orientation formula and B uses the bounded
reference; all other training settings are matched. The evaluation CLI adds
staggered yaw/exit timing and additive turn diagnostics. See
`docs/modules/turn_envelope.md` and the frozen run protocol under
`plane/outputs/turn_envelope_20260925_162526/` for responsibilities and
thresholds. Final `python3 tools/check_architecture.py` reports zero cycles and
layer violations (254 modules, 1083 edges); 671 training tests pass. A/B each
ran 64-env 1-iteration smoke and 4096-env 500 additional iterations. Three-seed
Isaac review retained all original 25 commands, but B did not produce a
meaningful active lean or expand the strict turn envelope. Read the run
`README.md` then `completion_report.md`; do not infer closed-chain acceptance.

# 2026-09-25 TURN_LEAN_LONG authorized ability exploration

This phase is a new, bounded training objective, not behavior-equivalent
refactoring and not a continuation of the 500-iteration A/B. The old
`TURN_ENVELOPE` recipe and its outputs remain intact. A new explicit spec
selects turn height/lean/cohort/LR; app validates exact source and the frozen
R10200 teacher, while the existing command scheduler now supports public
height ramps only when configured. The evaluator adds measured whole-body
COM, posture error and failure diagnostics without changing the old gate.
`docs/modules/turn_lean_long.md` maps entry points; live stage status and
bounded milestone summaries are under
`plane/outputs/turn_lean_long_20260925_175224/`. The phase is ongoing;
training scalars alone are not acceptance evidence.

# 2026-09-26 cornering height skill (authorized training behavior change)

This is not architecture equivalence. The exact model_35250 source and R10200
teacher are frozen by SHA in `plane/outputs/cornering_height_skill_20260926_122332/spec_stage1.json`.
The optional cohort plan adds a 20-slot 50/20/25/5 retention/height/mid/high
split and a public independent height scheduler; old recipes are unchanged.
The only reward change exempts `base_height` from per-term clipping because
the old 8x height term is flat at a 0.04m error. Stage 1 uses 0.38m practice,
1e-5 fixed actor LR, frozen encoder, exact full model/two-Adam restoration,
and at most 5000 new iterations; stage 2 is conditional, with total budget
<=20000 and at most two runs. Its status and milestone evidence are under
`plane/outputs/cornering_height_skill_20260926_122332/stage1_job/`.
The 64-env smoke restored all model/optimizer tensors exactly; its summary
showed reference fraction .5, encoder delta 0 and LR 1e-5. Architecture
audit: 261 modules, 1138 edges, no cycles/layer/missing imports. Training
tests: 682 passed, 2 warnings before the final sampler test was added;
the targeted turn tests subsequently passed 10/10. Source-tree motion
acceptance is pending and cannot be inferred from these code checks.

## 2026-09-26 turn-lean review height fix

The existing `app.turn_lean_review` previously sent `stage.turn_height` for
every turn command. In the cornering-height spec this value is 0.40m, so it
silently evaluated the wrong target for mid/high loads. Review now validates
the spec and derives each turn target from the ordered `height_schedule`
thresholds and `abs(vx*yaw)`; retention remains 0.40m and old fixed-height
specs keep their former target. A temporary-job fake evaluator test checks
actual CLI arguments without launching Isaac. Independent height-skill and
return-to-0.40m evaluation remains separate. No running trainer/monitor code
was changed by this fix. Full training tests: 684 passed, 2 warnings.
Architecture: 261 modules, 1142 edges, no cycles or layer violations.

## 2026-09-26 candidate evidence chain correction

The old untracked `screen_turn_lean_candidates.py` read one seed's
`turn_train.json` and looked for a nonexistent top-level `passed` field,
thereby counting every real retention/turn row as failure. It also treated
three neighboring files as a complete three-seed review without checking
checkpoint or protocol identity. The corrected entry consumes the existing
long summary, verifies raw metadata for all required group/seed files,
binds checkpoint SHA and actual iteration, and compares only identical
protocol/evaluator/gate/metric signatures and command sets. Missing, empty,
skipped, fixed-height and legacy-identity results are not accepted. New
summaries carry metric/gate source SHA; historical raw results can be
re-summarized to a separate v2 sidecar without rewriting them. The current
height-skill review adds independent 0.38/0.36m steady and exit/recovery
groups. Its outputs remain Isaac tree evidence, not closed-chain acceptance.

## 2026-09-26 bounded height-stage result and continuation

Stage 1 completed its 5000 added iterations at actual checkpoint40250. Five
1000-iteration probes retained the five basic commands but reported 0/6
independent height passes throughout; 0.38m mean root-height error stayed
about 0.019m and mid-turn passes fell from 2/8 to 1/8. The actor changed
8/8 tensors while frozen encoder changed 0/6; command0.38 appeared in both
current and latest-history policy channels. The old source35250 remains the
reviewed initialization; model40250 is not accepted.

The second and last authorized run restarts source35250 with 50% retention,
30% independent 0.38m practice, 15% mid turns and 5% existing high boundary.
The encoder is trainable at 3e-6 with actor LR1e-5. Its 64-env smoke verified
exact full model/two-Adam restore, reference fraction .5, actor and encoder
parameter changes, and unchanged policy contract. The formal run is bounded
at 15000 additional iterations; stage1+2 total cannot exceed 20000.
The monitor pauses after three consecutive 1000-iteration probes without a
height pass or at least 3mm height-MAE improvement, then uses the existing
completion hook. Runtime identity and status are in the current experiment
`source_identity.md` and `stage2_job/status.json`; no stage3 run is authorized.
