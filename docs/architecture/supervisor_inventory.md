# 监督器入口清单

当前所有列入架构审查的监督器实现位于 `plane/wheel_legged_gym/app/`，
`tools/` 的同名文件仅是 CLI 引导。导入不会启动 job；调用 `main` 才会执行。
逐入口的子进程、lock、STOP、status、PID、恢复、hash、通知和覆盖风险见
[process_review.md](process_review.md)。不能因文件在 app 内便假定具备这些机制。

| 任务 | app 入口 | 继续阅读 |
|---|---|---|
| 运动目标与闭链候选 | `motion_supervisor`, `candidate_closed_review` | `docs/modules/motion_supervisor.md`, `candidate_closed_review.md` |
| 高度训练与筛选 | `fixed_height_diagnosis`, `run_height_course`, `run_dual_height`, `screen_height_checkpoints`, `continue_height_course` | `docs/modules/fixed_height_diagnosis.md`, `height_supervisors.md` |
| 闭链序列与速度 | `run_closed_sequences`, `run_closed_stop_comparison`, `run_dynamic_boundary`, `validate_closed_ramp`, `validate_closed_speed` | `docs/architecture/process_interfaces.json`, 目标 app 源码 |
| 历史训练 | `run_encoder_ablation`, `run_h3_low_speed`, `run_stand_ablation`, `train_stand_long` | `docs/modules/historical_supervisors.md`, 目标 app 源码 |
| 适配与诊断 | `run_sim2sim_adaptation_pair`, `run_stop_ramp_diagnostic`, `validate_adaptation_candidates`, `validate_basic_dynamics`, `audit_observation_noise` | 目标 app 源码、对应 `plane/tests/` |
| 对比、完成与归档 | `compare_policy_versions`, `wait_for_completion`, `summarize_policy_comparison`, `export_model_registry` | 目标 app 源码、旧 job manifest/status 格式 |

历史迁移过程和冻结旧源码的等价测试见 `refactor_progress.md` 与
`plane/tests/test_supervisor_entrypoints.py`、`test_remaining_supervisor_entries.py`。
架构审查不要执行这些 CLI 或无 argparse 脚本的 `--help`；它们可能创建任务、
子进程或覆盖旧 job 摘要。只用 AST、导入测试、fake subprocess、临时目录核对。
