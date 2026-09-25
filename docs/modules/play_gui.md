# GUI播放状态边界

`wheel_legged_gym/scripts/play.py`是交互式GUI入口，不参与训练和headless验收。
`PlayCommandState`现在由每次`play(args)`创建，持有`cmd_x`、`ang_vel`、`cmd_height`、
按键标志、`running`、RLock及`runtime_limits`。pynput回调、web panel和仿真循环通过
显式绑定state的回调访问，不再共享模块级命令变量；训练domain不持有GUI状态。

相机/环境初始化仍在play入口，运行时环境变量仍只作为GUI操作参数，不进入训练domain。
迁移前需要GUI或假键盘测试覆盖：左/右互斥、同时按键归零、height上下限、退出标记、
lock释放和runtime_limits的更新。`test_play_state_boundary.py`检查session字段、回调绑定
和模块级状态缺失；当前不启动GUI，不把交互线程行为冒充为训练验收。
