# 监督器剩余范围清单

2026-09-24现场AST扫描tools下run/train/validate/screen入口，发现以下16个文件仍直接调用
subprocess.Popen。存在子进程调用不自动意味着设计错误；每个文件需要按职责判断，不能
把移到app目录本身当作完成。以下为待核对清单，不是允许循环/向上依赖的例外。

| 分组 | tools文件 | 需要核对的边界 |
|---|---|---|
| 高度课程 | run_height_course.py、run_dual_height.py、screen_height_checkpoints.py | 课程/候选/转换验收与进程、artifact分离；保留seed、遇错提前停止与两高度完整执行语义 |
| 历史训练 | run_encoder_ablation.py、run_h3_low_speed.py、run_stand_ablation.py、train_stand_long.py | 已用evaluation.gate的保留；独有决策提取，进程取消/恢复语义明确 |
| 闭链序列 | run_closed_sequences.py、run_closed_stop_comparison.py、run_dynamic_boundary.py | 命令序列与指标已有domain/evaluation；核对线程调度、STOP、hash校验与摘要落盘 |
| 诊断与适配 | run_sim2sim_adaptation_pair.py、run_stop_ramp_diagnostic.py、validate_adaptation_candidates.py、validate_basic_dynamics.py | 子job发现/恢复、动态门槛与任务状态隔离，不能隐式提升模型 |
| 闭链网格 | validate_closed_ramp.py、validate_closed_speed.py | ramp已有validation_schedule；核对剩余进程、逐方向停止和skipped统计 |

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
