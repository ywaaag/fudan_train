# Codex 工具索引

这是训练仓库唯一的“任务 -> 工具”索引。先读本页，再读目标模块指南；不要
通过扫描 `tools/` 猜入口，也不要为同一任务新建第二套工具。以下命令均从
`/home/kellen/fudan_train` 执行，Isaac 命令需先使用 `fudan_leg` 环境：

```bash
source /home/kellen/anaconda3/etc/profile.d/conda.sh
conda activate fudan_leg
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/kellen/fudan_train/plane
export CUDA_VISIBLE_DEVICES=0
```

## 快速映射

| 任务 | 唯一入口 | 安全性 / 副作用 | 失败后先读 |
|---|---|---|---|
| 查看训练效果 | `tools/summarize_training.py RUN --window 50` | 只读 TensorBoard；不启动训练 | run 的 `policy_experiment.json`、对应 event 与 stdout |
| 查看一个 run 的状态 | `jq ... plane/outputs/<job>/status.json` | 只读 | `status.json`、`progress.md`、对应 operation log |
| 等待终态 | `tools/wait_for_completion.py JOB --timeout 1800` | 只读轮询；不启动子进程、不发通知；终态通知由监控器/完成 hook 负责 | `status.json`、`completion_hook.json` |
| 长训阶段监控 | `tools/monitor_turn_lean_long.py --help` | `--help` 安全；实际运行会轮询 checkpoint、写摘要/探针、必要时 STOP 当前 child | 本轮 `README.md`、`status.json`、`summaries/`、`probes/` |
| 汇总固定高度/长训转弯 | `tools/summarize_turn_envelope.py --help`；长训加 `--long` | `--help` 安全；实际汇总读取原始 JSON 并写新 summary/heatmap，不启动仿真 | `protocol.md`、原始 JSON、失败轨迹 |
| 生成状态轨迹视频 | `tools/render_turn_trace.py --help` | `--help` 安全；实际读取 JSON 写 MP4，不启动仿真 | 输入 JSON 是否完整、ffmpeg 错误 |
| 运行候选评估 | `tools/review_turn_lean_long.py --help` | `--help` 安全；实际启动 Isaac 子进程并写评估 JSON；变高度转弯按 spec 的 `height_schedule` 逐点下发，新 `cohort_plan` 另测独立高度与回高 | checkpoint SHA、评估 log、原始 JSON |
| 降高/内倾候选筛选 | `python3 tools/screen_turn_lean_candidates.py REVIEW_DIR --out NEW_RESULT.json`；`--help` 安全 | 读取 `long_evaluation_summary.json` 与少量原始 JSON 元数据；只写全新结果，不启动仿真、不覆盖历史；单 seed probe 不能输入为完整 review | 协议/评估器/gate/hash、缺测、固定高度、保留回归和逐组失败原因 |
| 导出 ONNX | `cd plane && python export_onnx/export_onnx.py --help` | `--help` 安全；实际加载 checkpoint 并写 ONNX | checkpoint 路径/SHA、导出 stderr |
| 验证 ONNX | `cd plane && python export_onnx/verify_onnx.py --help` | `--help` 安全；实际加载 ONNX/运行 CPU 推理，只写 stdout | checkpoint、ONNX、batch 和误差 |
| 完成报告 | `tools/training_completion_hook.py --help` | `--help` 安全；`--report-only` 写本地报告；普通模式可启动独立 Codex 复盘 | job 的 `completion_hook.json`、`completion_review.json` |
| 架构审计 | `python3 tools/check_architecture.py` | 只读 AST；不导入 Isaac；`--write` 才写依赖图 | 输出中的 cycles/layer/missing imports |

## 精确命令

查看训练效果：

```bash
env PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/summarize_training.py \
/absolute/path/to/plane/logs/wheel_legged/<run> --window 50
```

checkpoint 摘要必须与 checkpoint 对齐时，加 `--through-iteration <checkpoint_iteration>`；
该选项只保留 TensorBoard step 小于该 checkpoint 的标量。

查看转弯长训：

```bash
env PYTHONPATH=/home/kellen/fudan_train/plane \
LD_LIBRARY_PATH=/home/kellen/anaconda3/envs/fudan_leg/lib \
/home/kellen/anaconda3/envs/fudan_leg/bin/python \
tools/summarize_turn_envelope.py --long \
--job /absolute/path/to/phase_review_or_long_job
```

该命令会写带 metric/gate 源码 SHA 的 `long_evaluation_summary.json` 和
`long_turn_heatmap.png`；目标目录已有同名文件时会拒绝覆盖。历史 review 如缺
指标源码身份，可加 `--summary-out /absolute/path/long_evaluation_summary_v2.json`
只写新的旁路摘要，不改旧 JSON 或图。长训的原始评估需先由
`tools/review_turn_lean_long.py` 生成，不能把 TensorBoard reward 当作转弯通过。

候选筛选只接受已有 `--long` 汇总的完整 review 目录；先对评估结果运行上面的
`summarize_turn_envelope.py --long`，再把该目录传给筛选器。
它要求 seed19/37/53、同一 checkpoint SHA、相同命令集合与协议身份；
新高度技能 spec 还要求 `height_skill` 和 `height_entry_exit` 两组。
旧摘要缺 metric/gate SHA 时标记 `legacy_metric_unknown`，不能与新结果直接比较；
用旁路摘要重新汇总原始 JSON 后才恢复可比性。
`pareto_candidates` 只是同协议比较，`recommendation=null` 表示没有唯一的已接受候选，
不应改用最新 checkpoint 填空。

完成通知的安全顺序：先使用 `tools/wait_for_completion.py` 等待已有终态，再读
`completion_report.md`。新任务启动前必须继承当前会话的 `HAPI_SESSION_ID`；没有该变量时只生成本地报告，不要猜测 session ID；不要运行
`--acknowledge-review`，除非当前会话确实完成了该 job 的复盘。completion hook 的
event_id、发送锁和 reviewed 去重由 `app.completion`/`adapters.notifications.hapi`
拥有，不要复制一套通知脚本。

## 明确禁止

- 架构审查中不运行 `train.py`、`tools/run_*.py`、`tools/continue_*.py`、历史监督器或无明确授权的 `review_*`。
- 不把 `tools/summarize_*` 的输出当作新验收；它们只汇总已有证据。
- 不执行 `tools/export_model_registry.py` 进行发现性检查；它会写固定 registry 数据。
- 不清理 `plane/logs/`、`plane/outputs/`、checkpoint、assets、`.deep-copilot/`。
- 不修改 `/home/kellen/fudan_rl_wheel_leg/plane` 或 `/home/kellen/wheel_leg_sim2sim` 来适配训练结果。

新工具只有在现有入口无法表达真实副作用或输入契约时才新增；新增后必须把命令、
解释器、输入输出、side effect、适用范围和失败定位补到本页，并增加入口测试。
