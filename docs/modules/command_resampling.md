# 命令重采样边界

公开入口为`domain.commands.resampling.resample`，只修改CommandBuffers中的选定行。
`commands`保存命令，`sample_mode`保存诊断类别，`segment_counter`决定轮换bank槽位。
范围、commands配置和设备均显式传入；模块不访问环境实例或仿真器。

```mermaid
flowchart LR
    ENV[LeggedRobot._resample_commands] --> RES[domain.commands.resampling]
    RES --> SAMPLE[domain.commands.command_sampling]
    ENV --> RESET[RandomStartStop.reset 延迟回调]
    ENV --> HEADING[Isaac torch_rand_float 回调]
```

顺序必须保持：采样水平命令→更新类别/段计数→随机高度→可选启停reset→height_bank高度覆盖
→可选heading随机采样。height_bank也保留被覆盖的高度随机抽样，否则后续随机序列会变化。
启停scheduler不能提前创建；应用回调仍在原reset位置获取它。
设备保持环境传入的字符串，Isaac TorchScript采样函数不接受torch.device对象。

环境负责决定哪些env需要重采样；domain负责采样与buffer更新。
测试`test_command_resampling.py`对照冻结的原环境方法，覆盖五种策略、heading及启停组合、
非选中行、随机数状态、reset时刻的buffer以及未知策略拒绝。
真实64环境/seed11/1iteration smoke证据见
`plane/outputs/architecture_refactor_20260923/resampling_smoke_equivalence.json`。

2026-09-25 的 `turn_envelope` 是单独授权的行为变更：`env_id%4==0` 的
转弯 cohort 使用 `domain/commands/turn_envelope.py` 开放命令调度，其他
75% 环境仍用原 `basic_motion` bank。其 callback 只在该策略执行；旧五种
策略的重采样顺序不变。验证和命令表见 `docs/modules/turn_envelope.md`。
