# app 动态进程边界审查（2026-09-25）

范围：`plane/wheel_legged_gym/app/` 下指定的 25 个入口；源码 AST、调用点、
`adapters/processes/` 和 `plane/tests/test_*entrypoint*.py`、
`test_python_job_runner.py`。所有指定模块在导入时均不执行任务；任务从 `main` 开始。
`tools/` 同名 CLI 为薄入口。下表的“有”表示源码具备该能力，不表示任意崩溃时都成功恢复。
没有运行任何历史训练或验证 CLI。

缩写：`J` 创建独立 job；`L` 独占 lock；`A` 临时文件后 replace 原子写 status；
`S` 检查/传播 STOP；`P` 记录并清理 child_pid；`W` 等待/回收子进程；
`R` 有显式恢复检查；`H` 核对输入/输出 checkpoint 或 policy hash；
`N` 可发通知。`-` 为无此机制或不适用，`委托` 表示通过进程适配器完成。
所有写新 job 的入口都可能覆盖 **同一 job 内** 的 status、operation log、临时结果；
新 job 通常用秒级时间戳，运行前必须确认目录不存在。以下“旧结果”列标明显式访问
既存 job 的情况；汇总/registry 入口会直接重写固定输出，运行前必须单独确认。

| app 入口 (`main` 参数) | J | L | A | S | P | W | R | H | N | 旧结果/用途 |
|---|---|---|---|---|---|---|---|---|---|---|
| `motion_supervisor(root)` | 有 | 有 | 委托 artifact | 有 | 有 | 委托 | 有 | 有 | 有 | 可恢复既存 job；训练与验收 |
| `candidate_closed_review(root, argv)` | 有 | 有 | 有 | 有 | 有 | 有 | - | 有 | 有 | 新 job；闭链验收 |
| `fixed_height_diagnosis(root)` | 有 | 有 | 有 | 委托 | 有 | 委托 | - | - | - | 新 job；训练及验收 |
| `run_height_course(root)` | 有 | 有 | 有 | 有 | 有 | 有 | - | 有 | - | 新 job；训练及验收 |
| `run_dual_height(root)` | 有 | 有 | 有 | 有 | 有 | 有 | - | - | - | 新 job；训练及验收 |
| `screen_height_checkpoints(root)` | 有 | 有 | 有 | - | 有 | 有 | - | - | - | 新 job；只读 checkpoint 验收 |
| `run_closed_sequences(root)` | 有 | - | 有 | 有 | - | 有 | - | 有 | 有 | 新 job；闭链验收 |
| `run_closed_stop_comparison(root)` | 有 | - | 有 | 有 | - | 有 | - | 有 | - | 新 job；闭链验收 |
| `run_dynamic_boundary(root)` | 有 | - | 有 | 有 | - | 有 | - | 有 | - | 新 job；闭链验收 |
| `run_encoder_ablation(root)` | 有 | 有 | 有 | - | 有 | 有 | - | - | - | 新 job；训练及验收 |
| `run_h3_low_speed(root)` | 有 | 有 | 有 | - | 有 | 有 | - | - | - | 新 job；训练及验收 |
| `run_sim2sim_adaptation_pair(root)` | 有 | 有 | 有 | 有 | 有 | 有 | - | - | - | 新 job；训练及闭链验收 |
| `run_stand_ablation(root)` | 有 | 有 | 有 | - | 有 | 有 | - | - | - | 新 job；训练及验收 |
| `run_stop_ramp_diagnostic(root)` | 有 | 有 | 有 | 有 | 有 | 有 | - | 有 | - | 新 job；闭链诊断 |
| `train_stand_long(root)` | 有 | - | 有 | - | 有 | 有 | - | - | - | 新 job；长训练，架构审查禁用 |
| `validate_adaptation_candidates(root)` | 有 | 有 | 有 | 有 | 有 | 有 | - | - | - | 新 job；验收/子 job |
| `validate_basic_dynamics(root)` | 有 | 有 | 有 | 有 | 有 | 有 | - | 有 | - | 新 job；闭链验收 |
| `validate_closed_ramp(root)` | 有 | 有 | 有 | 有 | 有 | 有 | - | 有 | - | 新 job；闭链验收 |
| `validate_closed_speed(root)` | 有 | 有 | 有 | 有 | 有 | 有 | - | 有 | - | 新 job；闭链验收 |
| `compare_policy_versions(root)` | 有 | 有 | 有 | - | 有 | 有 | 有 | 有 | - | 可恢复既存 job；checkpoint 验收 |
| `continue_height_course(root)` | 有 | 有 | 有 | 有 | 有 | 有 | 有 | - | - | 可续既存 job；训练及验收 |
| `audit_observation_noise(root, argv)` | - | - | - | - | - | 有（`subprocess.run`） | - | - | - | 启动诊断 CLI；写指定输出 |
| `wait_for_completion(argv)` | - | - | - | - | - | - | - | - | - | 轮询旧 job；只读等待 |
| `summarize_policy_comparison(argv)` | - | - | - | - | - | - | - | - | - | 读取旧结果并覆盖同 job 摘要/CSV |
| `export_model_registry(root)` | - | - | - | - | - | - | - | 有 | - | 核对旧 checkpoint hash，覆盖固定 `docs/data` 文件 |

这里的 `R=-` 指没有完整的显式恢复入口，不能根据残留 `status.json` 自动重启。
`H=-` 指该入口本身未核对 checkpoint 内容 hash；即使下游评估器检查 hash，仍须
另行核对。`S=-` 的入口不可假定 STOP 文件能安全中断。并发序列组的 `P=-` 意味着
status 中没有统一 child_pid；终止行为通过各 trial 的 Popen/terminate/wait 处理。
`train_stand_long` 还启动 completion hook 子进程，审查时不能执行其 CLI。
`wait_for_completion` 只轮询 status、completion hook 和 supervisor PID，不创建 job；
`summarize_policy_comparison` 只是把 manifest 中的 hash 写入报告，不重新校验文件。
`compare_policy_versions` 的显式 root 路径与旧 manifest 拒绝恢复已由临时目录测试验证；
新 job 的 `runner_sha256` 指向 app 源码，历史 job 仍按其原 hash 保留。
`export_model_registry` 的 CLI→`main(root)` 参数由受控 `runpy` 测试验证，
没有执行真实 `docs/data` 导出。`audit_observation_noise` 保留调用者 cwd，
相对 checkpoint/输出参数按 CLI 调用目录解释。

已有测试用 fake `Popen`、临时目录和导入检查验证部分启动边界；
`test_python_job_runner.py` 验证 STOP 后 terminate/wait 和回调顺序。
这些测试不能证明 25 个历史入口的任意故障恢复、秒级 job 名冲突、通知服务可用性，
也不能把保护拒绝解释成运动能力通过。第三方边界为 Isaac/PyTorch 进程退出语义、
MuJoCo CLI、操作系统 PID、HAPI 通知以及历史 job 文件格式。
