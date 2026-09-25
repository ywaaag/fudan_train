# 验收计算模块

一句话职责：对已经采集的结果计算门槛、响应与连续序列指标，不启动仿真或选择训练配置。

公开入口：

- `evaluation.gates.gate(row)`：保持原失败、接触、速度、高度和饱和度阈值。
- `evaluation.transitions.response_metrics(...)` / `summarize(data)`：稳定时间、超调与停车路径。
- `evaluation.sequences.summarize(data)`：显式序列的逐段跟踪和物理完成检查。

输入为已记录JSON兼容结构，输出为JSON兼容字典。输入协议、warmup及采样频率决定
结论范围；不能以稳态通过推断阶跃通过，也不能将没有测到的停车距离当成0。
门槛是行为契约：本次仅搬迁，不修改公式、比较符或通过条件。

依赖仅允许contracts/domain/evaluation及numpy等基础库。禁止导入tools、app、Isaac或MuJoCo。
运行环境与结果采集由适配器负责，命令选择和流程推进由工作流负责。
旧`tools/summarize_transitions.py`等仍是可用CLI，但新模块不得导入这些兼容入口。

验证：`PYTHONPATH=plane python -m pytest plane/tests/test_transition_summary.py plane/tests/test_policy_comparison_gate.py -q`。
修改前同时检查现有失败样本，不能为了新模型通过而放宽阈值。
