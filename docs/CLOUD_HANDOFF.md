# Codex 云端接续

## 2026-10-18：基于最新 main 的业务任务整合

先 fetch 并核对远端 `0e055e89b166896214e297f2324fb8e1f51628bb`，保留其执行器与优化流程，再将业务场景、1～100 问、明确问题标注及分业务报告整合为 `902e591`，已快进推送 main。旧分支 `codex/frozen-evaluation-protocol` 仅作历史保存，没有整套覆盖远端实现。开发前同步远端的规则已写入 AGENTS.md。

当前使用流程以 [USER_GUIDE.md](USER_GUIDE.md) 为准。349 项 pytest、6 项前端测试及合成 7 问/业务场景/明确问题模式的桌面与手机 Chromium 流程通过；pytest 有现有 Starlette/httpx 弃用警告。没有真实 GPU、模型下载/训练、付费调用、生产部署或凭据迁移。仓库推送不代表环境 Publish、新任务恢复或远端 CI 已验收；下文“50 问”“未推送”等是历史阶段记录。

## 2026-10-17：已授权代码发布与 CI 启动器修复

用户已明确授权把本轮代码提交推送到独立 kirin7002/kev_trainer。完整实现已以 `ab94c7abe811c48ad664583987bb1d9315573d84` 常规推送 main，远端核对一致，无强推；下列历史“未提交/未推送”描述是此前阶段状态。当前状态以 Git 记录为准。

首次远端 CI 的 320/321 项通过，唯一失败为启动器停止时无法确认身份；本地用 COLUMNS=70 复现。server.py 通过 ps -ww 读取完整 argv，保留原有 PID/启动时间/脚本路径归属检查；测试覆盖窄/宽列及拒停无关进程，没有跳过测试或放宽安全门禁。该修复单独提交，远端最终结果以 GitHub Actions 为准。

已取得实际页面桌面主页和手动流程截图，均为临时示例、GPU 关闭；截图不进入代码仓库，不部署公共预览。Library 保存若仍在准备前网络失败就报告原始阻塞与本地路径，不能伪造文件 ID。真实 GPU/模型/付费 Claude 仍未验收。


## 最新：baseline-only、镜像来源与上游证据复核

先读 EVIDENCE_AND_BASELINE.md / UPSTREAM_EVIDENCE.json。已实现实验级 gpu_baseline_v1 七阶段（不创建 Trial、不训练）和第二阶段基线固定引用；运行时自动观察已有 Podman 元数据中的 config/manifest digest，无法取得明确未知，不提升权限或连接 daemon。保持单槽/统一预算/停止确认和不覆盖旧好模型/基线。

已读取 GitHub 固定 commit/tree、15 个选定文本并校验 blob/SHA；base revision 取自固定 README 声明，archive SHA/大小取自 release API，权重没有下载/核算，Hub revision API 独立检查受 403 阻断。不把上游报告当作本平台实测。固定 train.py 472–473 拒绝 LoRA resume，平台也明确拒绝；重试只是从固定父 checkpoint 的新 attempt。

真实 GPU/依赖安装/缓存权重/业务数据、实际镜像 manifest digest、真实 Claude/服务商成本上限仍未验收。Library 如仍网络失败就报告真实错误，不绕过；旧五份补丁保留，新完整补丁另存 kev_trainer-baseline-evidence-review.patch。不推送、不部署、不新建认证或扩大第三阶段。


## 历史：独立执行复查与第二阶段软件

先读 OPTIMIZATION.md、MINIMUM_GPU_ACCEPTANCE.md、REQUIREMENTS_GAPS.md 与 VERIFICATION.md。新增 optimization.py/plan_validation.py、第二阶段/API/UI/测试；已实现严格 v2 方案/六输入/准确 SHA 与输入审批/串行候选/validation 选择/停止和持久保守费用预留。执行侧加固组停止确认、旧缓存/ZIP 晋升重验、策略一致、数据清单冻结和时钟回拨。不要丢弃现有改动或旧四份补丁。

真实 GPU/权重/依赖/业务数据，以及真实固定 Claude CLI/bare/账号/付费和提供方成本硬限额仍未验收；仅假 CLI、合成后端、真实本机临时进程/HTTP/Chromium。未调用真实模型、下载权重或连接远程。第三阶段未推进；未推送/PR/部署/创建凭据。后续应按最小材料清单做明确批准后的设备验收，不把软件夹具说成模型质量。


## 历史接续：第一阶段执行软件已实现

请先读 [EXECUTION.md](EXECUTION.md)、[REQUIREMENTS_GAPS.md](REQUIREMENTS_GAPS.md)、[VERIFICATION.md](VERIFICATION.md)。不要丢弃当前改动或三份旧补丁。新增 execution/model_artifacts/pipeline_validation/pipeline_commit/trial_execution/final_evaluation/kev_runtime/worker_execution，实现默认关闭的 GPU 13 阶段、统一预算、tokenizer/loss、独立 calibration/benchmark、loader/模型契约和独立 test-only 八阶段。worker_queue 使用单进程单槽原子 JSON 账本，训练/evaluator 令牌分离；不是 SSH 推送或生产多进程调度。

本轮只有合成契约后端、真实临时进程、本机 HTTP 和浏览器验收。没有 Torch/Kev 安装、模型下载、真实训练、GPU/远程连接、生产部署或持久凭据操作。真实设备/固定依赖/权重/tokenizer/张量/校准/loader/业务目录与数据仍独立阻塞。第二、三阶段没有宣告完成。源码 commit/archive/uv.lock 已锁定，不再把 CLI 未知列为阻塞；平台 custom calibration 明确使用固定 fitting API 加独立来源登记，见 EXECUTION.md。


## 历史接续：第一阶段准备链路

请先看 [REQUIREMENTS_GAPS.md](REQUIREMENTS_GAPS.md)、[ORIGINAL_DESIGN_HANDOFF.md](ORIGINAL_DESIGN_HANDOFF.md)、[KEV_CONTRACT.md](KEV_CONTRACT.md) 和 VERIFICATION.md 的最新章节。不要重建/丢弃工作区，不再把原设计或 CLI 源码未知当作阻塞；固定 Kev 源码已通过 GitHub 只读连接器取得并校验哈希。真实运行环境、权重、tokenizer、GPU 和上游校准来源登记仍未完成。

新增 phase_one.py、worker_queue.py、worker.py、worker_tasks.py、kev_contract.py 和 kev-source-contract.json：任务目录/原文件版本/五类分区及分组切分、手动不可变 Trial、拉取/心跳/租约与 CPU 进程检查、非模型准备包、validation 选择/test 门禁、全对象概率指标及不确定性。UI 已可使用；真实 Chromium 的桌面/窄屏验收通过。仍没有 GPU 训练执行器，不要将 CPU 作业改名为训练或宣称完整模型已产出。

后续优先补第一阶段 GPU 阶段执行器/统一时间预算、真实 tokenizer 和 smoke loss 观测、上游来源兼容的 calibration 注册、benchmark coverage 和真实 loader/模型包晋升。第二阶段自动 Claude trials/审查/计量及第三阶段 4B/多 Worker/部署未完成。坚持 Worker 拉取，不能改做 SSH 推送；原设计 PostgreSQL/Go/Vue/OSS 与当前单服务 JSON 约束的差异已记录，未擅自重写。

旧补丁 /workspace/scratch/kev_trainer-review.patch 和 kev_trainer-review-current.patch 保留；本轮原设计对照后的完整补丁另存，不提交、不推送、不部署。下面是早期迁移交接历史，其“尚未做 Linux/浏览器”等条目以最新验证为准。

2026-10-16 云端开发更新：已按锁文件建立临时 `.venv` 并复跑基线，继续完成输入边界、方案重试/取消、报告一致性和前端状态修复；新增真实 HTTP 运行验收和 CI 工作流草稿。当前改动保留在工作区，未提交、推送或开 PR。最新验证见 [VERIFICATION.md](VERIFICATION.md)，范围、缺口与后续验收见 [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md)。初始化通过不代表训练闭环完成，也不代表环境准备已 Publish。

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
