# 重构前已有改动的Git快照

2026-09-23 10:06:34备份保存了重构开始前的HEAD、diff和dirty/未跟踪文件。
本次从该备份重建独立Git提交，未用当前重构文件冒充原始状态。

| 仓库 | 归档提交 | 父提交 |
|---|---|---|
| fudan_train | a53e7df142558beb037461bd4b94ce189e713b46 | e5481e3cf0ff05917e24458d9eb5c8b930de5f1f |
| wheel_leg_sim2sim | 0b53547451548de5bcf8fec8143f0a52d9e8957b | ef91831f9d8abbcc0718c1437958aca459a67a31 |

两仓库分支名均为`archive/preexisting_20260923_100634`。它们不是当前重构成果提交。
可只读查看：

```bash
git show --stat archive/preexisting_20260923_100634
git diff archive/preexisting_20260923_100634 -- plane/wheel_legged_gym/envs/base/legged_robot.py
```

重建过程：独立临时index从备份HEAD开始，应用备份diff，再逐文件写入tar记录的源文件
及执行位；校验训练仓库60个、sim2sim仓库1个归档文件的Git blob与备份内容相同。
session日志没有纳入源码提交，仍保存在原tar及原目录，不删除或改写。
记录包括备份SHA256、树、父提交及排除文件，位于
`plane/outputs/architecture_refactor_20260923/preexisting_git_snapshots.json`。

建立归档分支前后验证HEAD不变、status输出不变、真实index字节不变。
后续整理当前重构提交时，以归档提交为对照区分用户已有改动与本轮重构；
不得盲目git add全部，尤其不要纳入session日志、模型和运行输出。
