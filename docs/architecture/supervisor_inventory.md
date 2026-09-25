# 监督器剩余范围清单

2026-09-24首次扫描发现16个tools入口直接管理子进程；阶段90/93已迁移训练/闭链主流程
至app同名main(root)，tools只引导路径。当前仍保留3个历史诊断工具直接调用子进程，另有
3个只读/完成hook工具维护旧job文件协议；它们明确列为后续边界，不伪称全仓库已无工具IO。
存在子进程调用不自动意味着设计错误；以下内部职责仍须逐项核对，不能把迁目录当作完成。

| 分组 | tools文件 | 需要核对的边界 |
|---|---|---|
| 高度课程（入口已迁移） | app.run_height_course、app.run_dual_height、app.screen_height_checkpoints；tools为薄CLI | 门槛已使用evaluation、配方使用experiments；仍核对进程/artifact边界，保留seed、遇错提前停止与两高度完整执行语义 |
| 历史训练 | run_encoder_ablation.py、run_h3_low_speed.py、run_stand_ablation.py、train_stand_long.py | 已用evaluation.gate的保留；独有决策提取，进程取消/恢复语义明确 |
| 闭链序列 | run_closed_sequences.py、run_closed_stop_comparison.py、run_dynamic_boundary.py | 命令序列与指标已有domain/evaluation；核对线程调度、STOP、hash校验与摘要落盘 |
| 诊断与适配 | run_sim2sim_adaptation_pair.py、run_stop_ramp_diagnostic.py、validate_adaptation_candidates.py、validate_basic_dynamics.py | 子job发现/恢复、动态门槛与任务状态隔离，不能隐式提升模型 |
| 闭链网格 | validate_closed_ramp.py、validate_closed_speed.py | ramp已有validation_schedule；核对剩余进程、逐方向停止和skipped统计 |
| 历史只读/完成hook（待收口） | audit_observation_noise.py、wait_for_completion.py、summarize_policy_comparison.py、export_model_registry.py | 旧job/diagnostic协议；迁移前先保留真实命令和报告格式，不能只删掉入口 |
| 历史高度续训（入口已迁移） | app.continue_height_course；tools为薄CLI | 保留round/seed/partial-candidate与STOP协议，需补边界测试 |
| 历史策略对比（入口已迁移） | app.compare_policy_versions；tools为薄CLI | 仍使用历史manifest/checkpoint协议，需在不运行长评估的前提下补边界测试 |

已明确分层的入口：run_motion_goal→app.motion_supervisor；run_candidate_closed_review→
app.candidate_closed_review；run_fixed_height_diagnosis→app.fixed_height_diagnosis；
四个历史导入即执行脚本→app同名main；summarize_training→app/adapter/evaluation。
这些仍需要最终总验收，但不应反复当作未迁移入口重新拆分。

下一轮优先顺序：高度组→闭链网格/序列组→历史训练/适配组。
每组完成应记录具体公开入口、数据依赖、失败/停止顺序和可执行测试；保留所有历史
实验参数。无argparse的历史脚本不能用--help探测，否则会真正运行任务。

高度课程本轮已提取`evaluation.height_acceptance.assess_height_row`：基于原gate，
micro高度误差上限.005，其余.015，严格大于才拒绝；非轮接触大于0追加full_contact。
原地替换row.gate、保留failed_checks顺序，seed补充由调用方负责。
test_height_acceptance覆盖36种stage/阈值边界/接触/基础门槛组合；其余高度流程未宣称完成。

阶段93的13个app入口：run_closed_sequences、run_closed_stop_comparison、run_dynamic_boundary、
run_encoder_ablation、run_h3_low_speed、run_sim2sim_adaptation_pair、run_stand_ablation、
run_stop_ramp_diagnostic、train_stand_long、validate_adaptation_candidates、validate_basic_dynamics、
validate_closed_ramp、validate_closed_speed。root及依赖root的PLANE/CP等路径为调用局部变量，
不再在模块导入时按__file__推算仓库位置；SOURCE等实验配置保持原值。

test_remaining_supervisor_entries用冻结旧源码检查main与辅助函数AST完全一致，
只排除root注入和路径常量归入main；禁止导入时解析参数、读写任务文件或Popen。
无argparse入口不执行--help。有argparse的8个入口在robot环境--help通过。
公开辅助函数audit_grid归app.run_h3_low_speed，旧测试调用方已迁移，不保留tools导出别名。
