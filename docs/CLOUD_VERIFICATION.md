# 云端开发验证

日期：2026-10-18。现有 Codex 云实例，Linux x86_64、Python 3.12.14。

## 本轮结果

- 完整 `.venv/bin/python -m pytest -q`：177 passed，1 条 Starlette/httpx 已知弃用警告。包含隔离数据、假 CLI 和真实本机启动器测试，没有真实模型调用。
- Chromium 页面流程：创建独立 car 业务实验、验证基线先行、导入候选、显示分业务指标、锁定候选、开放最终测试。页面无 JavaScript 异常。
- 桌面 1440 像素、移动端 390 和 360 像素页面检查通过，没有页面横向溢出；表格和代码块允许内部滚动。
- 真实启动器 `/api/health` 为 HTTP 200，`machine.connected=false`、`training_execution=not_connected`、Claude CLI 不可用。
- UI 预测是明确构造的测试文件，不是模型推理或效果证据。测试状态位于临时目录，不进入仓库。
- 锁定的上游 Kev 源码已核实 `--data`、`--init_from`、LoRA/head 参数及自定义数据长度过滤逻辑；没有安装或执行该上游训练程序。

## 仍未验收

真实基线推理、GPU/SSH/WSL2、训练执行器、参数更新、检查点重载、真实候选推理、跨进程适配器切换及业务效果。固定协议仅校验外部预测的声明和完整性，不证明其实际模型或推理配置。没有发布环境快照，也没有验证发布后新任务恢复。

原 `docs/VERIFICATION.md` 保留迁移前本地记录，不作为本轮云端证据。
