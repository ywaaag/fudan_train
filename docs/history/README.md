# 根文档历史快照

README_before_architecture_20260923.md与AGENTS_before_architecture_20260923.md
保存整理前的根文档原文，包含此前已有dirty内容和历史训练说明，不作现行操作指令。
这些快照中的相对路径/链接按原仓库根解释，原文中的“最新/当前”只代表记录当时。

当前代码入口、职责与操作约束请返回仓库根README.md、AGENTS.md和ARCHITECTURE.md。
快照不更新、不作为模型验收或进程存活证据。

`export_encoder_jit_135.py.txt`保存已移除的135D历史encoder一次性导出脚本原文。
它使用27D×5历史输入和写死的/root/gpufree-data路径，且导入即读checkpoint/写JIT，
仓库内没有调用方；按用户“不必兼容旧用法”的要求移出运行入口，文本仅供追溯，不执行。
当前25D/125D/6D部署策略使用plane/export_onnx/export_onnx.py及verify_onnx.py。
