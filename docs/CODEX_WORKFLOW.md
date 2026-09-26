# Codex 训练工作流

按此顺序执行：现场快照 -> 最小阅读 -> spec/protocol -> smoke/manifest -> 长训监控 -> 里程碑 -> 候选筛选 -> ONNX/终态。工具解释器和副作用见 `docs/CODEX_TOOL_INDEX.md`。

1. 现场：运行 Git branch/log/status、进程/GPU 查询和 `python3 tools/check_architecture.py`。记录已有进程，不杀进程、不抢 GPU。
2. 阅读：只读 `AGENTS.md`、`docs/CODEX_QUICKSTART.md`、工具索引、目标模块指南、run manifest/status/最新摘要；不要先打开全部 stdout/event/逐步 JSON。
3. 训练前：新目录写 `spec.json`/`protocol.md`，冻结 source/teacher SHA、contract、seed/env、LR、encoder、reference、bank、holdout、height/lean 公式、gate、预算和停止条件。
4. Smoke：先运行目标 recipe `--help`、目标 pytest、64 env/1 iteration smoke；核对 manifest、完整模型/Adam、reference、LR、encoder、contract 和 NaN，失败即停止。
5. 长训：记录实际 run/PID/完整命令，绑定 `tools/monitor_turn_lean_long.py`。每1000新增 iteration 写摘要/探针/status，每250保存 checkpoint；分开记录 target_iteration 和 checkpoint_iteration。
6. 里程碑：每1000步轻筛，每5000步或课程升级前阶段评估。优先原25命令、reset/接触/限矩、目标高度/COM、roll/pitch/关节余量、vx/yaw/slip/lateral velocity/power、holdout/动态出弯。不要按 reward 或视频选模型。
7. 候选：先单 seed，再少量候选运行 `tools/review_turn_lean_long.py` 的 seed19/37/53 评估，用 `tools/summarize_turn_envelope.py --long` 汇总，再用 `python3 tools/screen_turn_lean_candidates.py REVIEW_DIR --out NEW_RESULT.json` 比较完整同协议候选。要求全 seed、全环境、旧 gate 和新 gate 分开报告；失败/未测点不插值，保存 checkpoint SHA。旧摘要无 metric/gate SHA 时先写 v2 旁路摘要，不能默认推荐最新 checkpoint。
8. ONNX/终态：最终候选才导出并做 batch256 验证。终态读 status、原始评估、汇总、失败分类和 ONNX log，再写 `completion_report.md`。

新训练和长时间评估必须继承当前会话的 `HAPI_SESSION_ID`。完成 hook 或转弯监控器
写入本地终态报告后，会向该 session 发送一次通知，只唤醒当前 Codex 会话，不启动
新训练。启动前检查：

```bash
test -n "${HAPI_SESSION_ID:-}" || echo "缺少 HAPI_SESSION_ID：只能生成本地终态报告"
```

不要猜测或硬编码 session ID。缺少变量时保留本地报告，不要为补发通知重启训练。
9. 收尾：重新运行架构审计、完整 pytest、`git diff --check` 和 `git status --short`；只提交明确归属文件，不 `git add -A`，不提交模型/outputs，不 push。

长训停止条件：NaN、策略 contract 漂移、保护异常、原能力持续崩溃，或预算/iteration 上限。连续3000步无姿态/高度/包络收益时读取阶段摘要，最多做一次明确配置调整；不要默认最新 checkpoint 最好。
