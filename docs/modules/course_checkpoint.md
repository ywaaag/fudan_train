# 课程checkpoint边界

`domain/commands/checkpoint_state.py`负责method_v1 payload、segment counter恢复、
staged payload及stage合法性检查；不读文件，不引用runner或Isaac Gym。
公开入口为method_state、restore_method_counter、staged_state、restored_stage。

环境的get_checkpoint_state/load_checkpoint_state保留兼容接口，并按旧顺序处理分支：
method_v1优先；否则只在启用staged课程时恢复。method counter形状不匹配仍保持原值，
pipeline不兼容仍拒绝；staged恢复仍只保存stage/pass_streak/last_metrics。
窗口统计和last_check_step没有新增持久化，不在本次重构中改变续训语义。

恢复stage后，环境依次赋值pass_streak和metrics，再重置窗口、应用范围。
这些有序状态变化仍由环境拥有；domain不接收整个环境对象，不执行隐藏回调。

测试`test_course_checkpoint.py`覆盖字段集合、counter往返、空状态/错误形状、pipeline
拒绝、stage范围和metrics引用语义。课程窗口累计仍待迁移，不因本模块完成
而宣称整个课程生命周期已解耦。

## 推进与日志

`domain/commands/staged_progress.py`的window_is_due判断检查间隔/episode数，
advance_stage计算stage/streak/是否升阶，log_metrics生成原TensorBoard字段。
这些函数不持有环境或窗口，不写命令张量。环境仍负责收集stats、保存last_metrics、
last_check_step，并在升阶后应用范围，最后清空窗口。

最终stage仍累积通过streak，失败仍清零，保留原实现。
`test_staged_progress_equivalence.py`对照冻结旧环境方法，覆盖六种课程边界和回调顺序。
测试针对有效课程配置，不代表训练课程本身已通过运动验收。

## 窗口累计

`curriculum_window.create_window`创建独立标量累计字典；accumulate_window接收已筛选
valid_ids、timeouts和CommandMetrics，只更新窗口。环境仍先过滤episode_length>0，
空集合直接返回，先增加episodes计数再累计，保留reset时序。
reverse_command_sum取绝对值后相加；forward/yaw沿用各自旧统计字段，不统一转换。
test_curriculum_window.py验证索引选择、空集合、独立窗口及源buffer不变。
