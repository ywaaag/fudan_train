# 2026-11-01 用户授权的历史产物清理

本次明确授权覆盖旧文档中 logs/outputs/checkpoint 的一般只读限制，仅用于清理历史产物；不授权新训练、仿真或修改配方。

仓库占用从约24.29 GiB降至8.70 GiB。保留777个完整checkpoint文件（712个不同SHA），删除1,390个无引用中间checkpoint。实际来源、原JSON明确引用的模型、各run最终完整保存点及ONNX/已评候选均保留；最后保存点不自动代表best。完整网络、两个优化器和checkpoint已有课程/环境状态未裁剪。

359个run的训练配置、原生效参数、奖励公式/实现源码快照、source/resume关系、seed、终态/预算、TensorBoard及既有记录保留。20,485份配置、源码及其他记录前后SHA一致。已有dirty changes保留，未提交Git。

319份大型评估JSON无损压缩为原路径加`.gz`：原约18.21 GiB的数据块压缩为约4.46 GiB，完整解压SHA逐份核验后才移除原JSON。完整矩阵、逐条件/逐环境、geometry trace及失败记录均在原文内。原工具只认JSON路径时，需要先恢复指定文件；不能把压缩记录的原路径缺失误读为这次删除了结果。

精确模型清单、删除索引、原路径/SHA及压缩映射、环境清单和逐run复现索引位于：

- [完整清理报告](/home/kellen/disk_usage_audit/2026_11_01_cleanup/REPORT.md)
- [保留模型及SHA](/home/kellen/disk_usage_audit/2026_11_01_cleanup/FUDAN_KEEP.tsv)
- [复现记录索引](/home/kellen/disk_usage_audit/2026_11_01_cleanup/fudan_reproduction_index.json.gz)
- [原始评估归档索引](/home/kellen/disk_usage_audit/2026_11_01_cleanup/fudan_archive_index.jsonl)

核验一个归档（不写文件、不仿真）：

```bash
python3 /home/kellen/disk_usage_audit/2026_11_01_cleanup/restore_fudan_record.py plane/outputs/motion_goal_20260922_125637/initial_start_stop_seed53.json --verify-only
```

去掉`--verify-only`会将指定原JSON恢复到原路径，占回相应空间；文件存在时拒绝覆盖。可替换为归档索引中的任一原始绝对路径或仓库相对路径。`.gz`继续保留。

本轮仅执行CPU读回、SHA/归档验证和架构静态检查，未通过新仿真证明逐位重现。已删除的中间checkpoint只有身份和配置记录，不声称其权重仍可恢复。
