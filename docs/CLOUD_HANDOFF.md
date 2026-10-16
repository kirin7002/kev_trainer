# Codex 云端接续

用户要求将开发迁到 Codex 云端，后续由 Codex 直接完成，不再委派 Claude Code。目标公开仓库为 `kirin7002/kev_trainer`，GPU 连接继续预留。

## 环境准备

在 Codex 中选择 Work in → Cloud → Create environment，选择本仓库，使用 Python 3.12+。安装命令：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
```

开发与测试不需要 GPU、模型权重、Claude Code 登录或任何私钥。网页可用 `./scripts/start.sh` 启动，默认 127.0.0.1:8091；`GET /api/health` 返回平台与未接入状态。

请完成环境检查并 Publish，再从该环境启动云端开发任务。当前本地聊天不能通过 host handoff 直接原地变为 Codex 云端任务；应在已发布环境中新建接续任务。

## 已有代码

- app.py：数据集、实验、后台方案、离线训练状态、预测导入与报告 API。
- datasets.py：多种格式、显式标签、冲突校验和分组留出验证。
- claude_runner.py：产品功能的真实 CLI 包装，结构化方案、日志、取消和超时。无测试集上下文或工具权限。
- evaluation.py：完整 ID / SHA 校验、总体与分组集合指标、耗时可比性。
- machine.py：未接入的机器接口，不执行远程操作。
- storage.py：磁盘 JSON 与原子写入。
- templates/index.html：已实现中文单页，包含数据预览、实验、方案、日志、预测报告和下载。
- server.py、scripts/：单进程启动与归属校验后停止。
- tests/：只使用临时数据及假 CLI 的测试。

## 已做的验证

本地已验证数据导入、实验创建及外部预测报告，手算校验了单/多对象准确率、F1 与 P95。浏览器检查了示例导入、预览、模型选择、实验创建和窄屏布局。未执行真实模型训练或 GPU 连接；未通过产品页面验证真实 Claude 方案调用。

迁移前最终自动化结果写在 `docs/VERIFICATION.md`，云端应重新运行，不把本地结果当成云端环境验证。

## 云端接续指令

继续完成独立 Kev Trainer 项目。先读 AGENTS.md、README.md 和本文件。所有开发由 Codex 直接完成，不再委派 Claude Code。保持一个 Python 服务与中文单页，GPU / SSH / Windows WSL2 接入留待用户后续测试。先在云端运行全套测试及健康检查，再继续审查输入验证、任务恢复/取消、数据隔离、报告正确性和界面可用性。修复实际发现的问题，完善 CI；不要自动调用真实模型、连接服务器或执行训练。测试不通过不得假报完成。产品仍保留 Claude Code 方案生成能力，若云端未安装或未认证，明确显示不可用，不替换成伪造方案。最终给出改动、测试证据与尚待真实 GPU/Claude 联调的事项。

## 不会迁移的内容

本地 Python 虚拟环境、CLI 登录、SSH 私钥、上传数据、实验状态与运行进程都不进入仓库。需要真实集成时由用户在云端单独配置认证、网络和机器权限，不能从本地凭据目录拷贝。
