# Kev Trainer development

- 开发前先检查工作区并 fetch/pull 最新远端 main，阅读新提交及交接文档，再做代码审查和范围判断。保留未提交工作，不能直接覆盖或强推；基于过期快照的功能须先与远端对照，避免重复实现。
- 用户最新要求：后续所有开发、修复、审查和测试由 Codex 直接完成。不要委派或调用 Claude Code 来开发本项目。
- 产品中的 Claude Code 方案生成集成仍保留；开发测试使用临时假 CLI，不调用真实收费模型，不复制本地认证信息。
- 独立项目，不依赖或修改 jev_demo，也不要访问用户的其他工程、SSH 配置或服务器。
- 保持一个 FastAPI 服务、中文单页、文件/JSON 存储；不要擅自增加数据库或复杂调度。
- GPU/SSH/WSL2 接入尚未完成，本阶段不连接远程机器，不启动训练。离线时必须明确显示未接入，不能产生假的训练、模型或性能结果。
- 最终测试集不能进入优化 Agent 上下文。外部预测报告必须注明来源与 training_verified=false。
- 运行环境 Python 3.12+、Linux/WSL2/macOS；开发安装 requirements-lock.txt。
- 验证命令：`.venv/bin/python -m pytest -q`。测试必须隔离数据目录和外部进程。
- `.state/`、`.run/`、`.venv/`、任何凭据和真实数据不提交。
