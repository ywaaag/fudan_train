# 架构目标验收台账

本页按原目标记录证据与缺口，不替代源码核查，也不把阶段测试通过等同于goal完成。
2026-09-24用户已取消旧用法兼容硬约束：完成调用方迁移后删除冗余旧入口，记录新API；
物理/训练/验收行为等价及成果保留要求不变。下表兼容项按这一最新要求验收。
阶段改动详情和确切文件见refactor_progress.md；所有模型/资产/旧结果保留。

| 原目标 | 已有证据 | 尚需完成 |
|---|---|---|
| 仓库自描述、最小阅读上下文 | 两仓库ARCHITECTURE、模块指南、静态依赖JSON；根README/AGENTS分离历史快照；本轮本地链接已核对 | 最终目录树和边界核对，确保指南与最终实现一致 |
| 自有代码循环清零 | 2026-09-24最终静态检查：训练243模块/1027边、sim2sim39模块/78边，均cycles=[] | 动态进程依赖仍需最终汇总；不再有静态循环缺口 |
| 单向分层 | 受管层级检查通过；envs不依赖utils；已迁移tools/scripts未反向导入其他tools或核心入口 | 3个历史诊断tools及3个只读/job工具仍保留旧IO协议，详见supervisor_inventory；需最终复核 |
| 环境模块化 | 奖励、观测、控制、终止、物理reset、资产/actor/索引/随机化、Gym张量、课程窗口、命令重采样、DOF/PD初始随机化及原点布局已分离；environment_lifecycle指南记录调度与状态所有权 | 最终入口与代表性运行验收；调度本身归环境，不机械拆成代理 |
| 工作流模块化 | motion评估、完成报告、状态存储、进程执行、验证调度、实验配置构造、轮次推进及恢复检查分离；高度附加门槛归evaluation；16个监督器入口已归app | supervisor_inventory.md中的应用内部进程/artifact职责/取消/恢复仍须核对，不以迁目录代替完成 |
| 消除隐式全局状态 | registry由app工厂创建；配方不读环境/文件；具名指标buffer无重复别名；类配置容器按实例隔离；Torch JIT显式app调用；MuJoCo guide配置显式输入；FUDAN_SCALES/TERMINAL只读；未使用的135D导出脚本归档；GUI命令状态归PlayCommandState | 最终扫描模块初始化及第三方运行时边界；Torch开关边界见configuration指南 |
| 跨仓库公开接口 | 三个CLI的process_interfaces.json；closed/tree冻结快照；训练导入边界和probe快照测试；旧after_step已删除 | 最终验收再扫描调用方与文档 |
| 单一公开入口 | 已迁移调用方并删除rsl_rl、旧env配置/配方及utils兼容包装；16个监督器归app；MuJoCo测量统一公开模块调用；tools入口扫描无旧监督器互导 | 最终核对公开入口清单；不要求旧用法兼容 |
| 训练行为等价 | 多次64env/seed11/1iteration checkpoint与两组optimizer精确一致 | 最终代码完成后的统一smoke和有代表性分支核查 |
| ONNX契约等价 | 当前final_reexport_10200.onnx与历史文件SHA256完全相同；batch1/8/256输出精确相同；PyTorch batch256误差7.63e-6通过；证据final_onnx_equivalence.json | 此后若改learning/导出路径需重验；最终汇总引用hash，不能以此证明所有模型运动能力 |
| sim2sim等价 | tree34000步；当前closed正常2000步与历史保护拒绝2489步全字段精确对照（仅排除耗时及源码hash），证据closed_success_rejection_equivalence.json；外部24项测试通过 | 此后若改验证路径需重验；最终汇总引用证据，不能将拒绝复现说成模型能力通过 |
| 版本可追溯 | 独立refactor分支、修改前备份、run完整源码快照；原dirty源码独立归档（git_preservation.md）；阶段89提交当前源码状态，提交日志可追溯 | 阶段提交不是最终验收完成；后续修改仍须单独验证/提交，以归档区分原dirty改动 |

完成条件是上表剩余项逐项有证据关闭。未解决事项不能用新增例外掩盖；确属第三方兼容
边界的例外须写原因、调用边界和移除计划。不得启动长训练代替架构验证。
