# 验证记录

## 2026-10-17：main 发布复验与窄终端停止回归

完整实现以 ab94c7abe811c48ad664583987bb1d9315573d84 推送 main，Git 远端完整 SHA 核对一致；未强推、部署或启动 GPU。此前未提交/未推送章节保留为历史证据。

首次 GitHub Actions 安装锁定依赖通过，后端 320 passed/1 failed：[运行记录](https://github.com/kirin7002/kev_trainer/actions/runs/38994788721)。失败为 tests/test_server.py 的停止归属核验。以 COLUMNS=70 本地复现后，增加 ps -ww 防止长脚本路径被截断；继续校验原 PID 身份与脚本路径，不停止无关进程。窄/宽列和无关进程三个定向回归通过。修复后全量本地 322 passed（50.04s）、前端 5 passed；远端 CI 结果以 Actions 为准，不把本地通过冒充远端通过。

实际 Chromium 再次通过 1440/390px、JS 错误 0；新增桌面主页和手动流程截图在 /workspace/scratch/kev-trainer-user-ui/，示例数据、未接入 GPU，未发布站点。截图经人工视觉检查没有凭据/个人信息，Library 上传受工具服务网络阻断，没有生成 Library ID。


## 最新：独立基线、镜像来源和证据分级（2026-10-16 UTC）

- `.venv/bin/python -m pytest -q`：**321 passed、0 failed、0 skipped，50.28s**。新增 18 项 baseline/image/resume 回归，首轮有一处测试括号语法错误已修正后运行通过；其余 303 项保留。
- baseline-only 全链真实临时进程/本机 API（合成模型后端）通过：不创建 Trial、全阶段无 train、数据只有 calibration/validation、共享预算、失败不替换旧好基线、租约过期/取消、controller 训练声明/策略篡改拒绝、自动方案安全基线来源。
- 镜像测试覆盖缺失/坏值/链接/可写/超大/错误 engine、config 与 manifest digest 分开记录、环境变量不能作证明、注册与运行来源变化拒绝晋升。当前环境实际采集返回 unknown/null；没有实际镜像 digest 验收。
- LoRA resume_from 在导入训练库前即拒绝；固定上游 train.py 已读，472–473 明确只对 full_ft 支持 resume points，本轮不实现 full_ft 或半恢复。
- Node 前端 **5 passed**；真实 Chromium 154.0.8037.57、1440/390px、JS 错误 0，原流程和第二阶段审查/等待资源/取消通过，新增基线按钮默认禁用断言。证据目录 `/workspace/scratch/kev-trainer-browser-baseline/`。没有真实 GPU/UI 结果或收费调用。
- pip check、py_compile、启动/停止 shell 语法、git diff --check 通过。既有 Starlette/httpx 弃用警告保留。
- 上游证据：15 个选定文本 hash 对照；uv.lock 335387 字节/SHA a9922dbb…；源码 README 明确完整 base revision，release API 明确 archive digest/size。权重未下载/核算，锁中的训练依赖未安装，Hub revision API 受 403 阻断。详见 EVIDENCE_AND_BASELINE.md / UPSTREAM_EVIDENCE.json。

新完整补丁 `/workspace/scratch/kev_trainer-baseline-evidence-review.patch`，保留旧五份；干净根提交 git apply --check / 应用后文件字节比对，哈希和大小另存 .patch.sha256/.patch.json。仍无提交、推送、PR、部署、网络权限修改或持久凭据创建。


## 历史：独立复查与第二阶段软件（303 项，2026-10-16 UTC）

仅独立 kev_trainer，基线 `573b0d4ba3490c9d74d8ab08ff89297d346bff8c`。未推送、部署、使用真实付费模型/GPU、下载权重、连接远程或创建凭据。

- `.venv/bin/python -m pytest -q`：**303 passed、0 failed、0 skipped，47.48s**；新增第二阶段和复查回归 **34 项通过**：方案语义/参数、六输入无最终 test、准确审批、串行两候选与 validation 选择、停止条件、失败/取消、幂等重放、方案篡改、GPU/Claude 预算、跨会话/重启不重置费用、假 CLI version/help/bare/成本回执、可信 validation 基线、导出策略不一致、缓存与 ZIP 篡改/晋升重验、无法确认停止不 ACK、时钟回拨和清单协调修改拒绝。
- 修复生成状态竞态，并重跑对应 34 项通过；一次中间测试曾读到尚未完成的失败标记，现不再发布该中间状态。没有将失败记录隐藏成首次全绿。
- CPU 取消测试将 monkeypatch 收窄到 worker_tasks 子进程，避免把用于停止核验的 ps 也替换成睡眠进程；两个真实 CPU 取消/失租测试 **2 passed / 0.38s**。产品 ps 查询增加 2 秒超时，无法确认保持占槽。
- Node 前端 **5 passed**，内联 JS 语法/乱序响应等保留。
- 真实 Chromium 154.0.8037.57 完成原准备/validation/test 外部报告，以及新增的 **方案导入→人工审核准确 SHA→开始→WAITING_RESOURCE（GPU 默认关闭）→取消**。1440/390px，JS 错误 0、无横向溢出。
- 新浏览器证据：`/workspace/scratch/kev-trainer-browser-optimization/`，额外 optimization-desktop.png / optimization-mobile.png；原截图保留。全部合成数据、临时认证和临时本机服务，training_verified=false。
- pip check、py_compile、启动/停止 shell 语法和 git diff --check 通过。既有 Starlette/httpx 弃用警告未改依赖规避。

本轮完整补丁 `/workspace/scratch/kev_trainer-optimization-review.patch`，旧四份补丁保留；干净根提交 git apply --check 和应用后逐文件字节校验，哈希/大小另存同目录 .patch.sha256/.patch.json。所有改动仍在工作区，远端 CI 未运行。

具体实现与真实验收边界见 OPTIMIZATION.md 和 MINIMUM_GPU_ACCEPTANCE.md。测试后端不是训练或财务账单证据；硬件/实际依赖与固定 Claude 账号调用仍独立阻塞。


## 历史：第一阶段执行软件（269 项，2026-10-16 UTC）

基线 `573b0d4ba3490c9d74d8ab08ff89297d346bff8c`，仅 kev_trainer。真实模型/GPU 验收未完成；[EXECUTION.md](EXECUTION.md) 记录实现和未来操作前置条件。

- Linux/Python 3.12.14，锁文件平台环境；本环境没有 Torch/Kev，没有下载权重或启动真实训练。
- `.venv/bin/python -m pytest -q`：**269 passed、0 failed、0 skipped（38.42s）**；新增 `test_gpu_execution.py` **28 项通过**，包含 13 阶段训练+8 阶段最终评测的合成契约、本机真实 HTTP ZIP 上传、角色/test 隔离、阶段失败/不晋升/保留旧模型、共享超时、实际进程组子孙取消、独立 watchdog、租约过期/旧上传拒绝/预算累计、worker 重连/恢复幂等、校准标签/coverage/阈值/loader/冻结策略篡改拒绝、ZIP 路径/重复/哈希/不可覆盖、loss observer 保留 tensor/拒绝丢样本、token 计数与环境凭据隔离。
- 合成后端仅测试依赖注入启用，所有模型/性能为夹具；始终 `software_contract_fixture/training_verified=false`，真实模型下载返回 409。真实 HTTP/临时进程成功只证明软件行为。
- Node 前端 **5 项通过**；整页内联 JS 语法检查通过。
- 真实 Chromium **154.0.8037.57**，CPU 准备→validation 选择→外部 test 报告与下载再次通过，**1440/390px、0 JS 错误、无横向溢出**。执行入口、预算、取消与来源显示已更新；没有用浏览器合成结果声称真实 GPU UI 验收。
- 证据 `/workspace/scratch/kev-trainer-browser-execution/`：evidence.json、manual-desktop.png、results-desktop.png、results-mobile.png、fixture-report.json。
- pip check 无依赖破损；py_compile、启动/停止脚本语法、git diff --check 通过。一个既有 Starlette/httpx 弃用警告保留，未改锁定依赖消除它。

完整补丁 `/workspace/scratch/kev_trainer-execution-review.patch` 包含相对根提交的 tracked 和新增文件，旧三份补丁保留。补丁在干净基线执行 git apply --check；补丁哈希/大小放同目录 .sha256 文件，避免将自身哈希嵌入补丁。未提交/推送/PR/部署，远端 CI 未运行。


## 历史：第一阶段准备链路（241 项，2026-10-16 UTC）

独立仓库 `kirin7002/kev_trainer`，基线 `573b0d4ba3490c9d74d8ab08ff89297d346bff8c`。原始初步设计摘要已由父任务补充并保存为 ORIGINAL_DESIGN_HANDOFF.md；需求和分阶段缺口见 REQUIREMENTS_GAPS.md。**这是可审阅的第一阶段准备实现，不是完整训练产品验收。**

### 最终自动化

- Linux / Python 3.12.14，按 requirements-lock.txt 的临时 `.venv`；没有发布环境安装/start 草稿。
- `.venv/bin/python -m pytest -q`：**241 passed、0 failed、0 skipped**；包括原有 151 项及新增输入/生命周期/数据分区/Trial/Worker/源码契约/评估/真实 HTTP 回归。
- `node --test tests/frontend.test.cjs`：**5 passed**，整页内联 JavaScript 语法及乱序响应/特殊标签与 ID。
- `.venv/bin/python -m pip check`：No broken requirements found；`bash -n scripts/start.sh scripts/stop.sh`、`git diff --check` 通过。
- 仍有 Starlette TestClient 对 httpx 的一个弃用警告，非失败；没有为消除警告改变锁定依赖。

### 真实本机服务、Worker 和进程

`tests/test_http_acceptance.py` 用临时项目/数据/端口，实际启动/停止脚本、真实 FastAPI 和 HTTP，完成原有示例报告/重启恢复以及新流程：50 对象合成任务、四类分区、手动 Trial、入队 CPU 准备作业、实际参考 Worker CLI 拉取并启动固定 CPU Python 子进程、上传结果与准备 ZIP 下载；训练请求仍返回真实 409 阻塞。

`tests/test_phase_one.py` 验证原文件字节和 SHA、全 50 对象 System One、原始 `_meta` 身份、固定目录/硬标签、训练池按关联组派生三分区、跨区表面近重复/显式组拒绝、未知对象/缺失 false 拒绝、配置/候选预算/代码启动指纹、确定性 ZIP 逐文件哈希、Worker 输入无 test/regression、独立认证且 token 不落盘、单槽租约与旧 owner fencing、服务重建后等待旧进程停止、真实临时进程取消/失租、完成提交时取消竞态、validation 先选候选/test 后验收、重复测试转回归和改 ID 不能回流训练、严格 test 预览不含逐条内容。

所有模型、目录、概率、预测和身份均为临时合成夹具；实际执行的是 CPU 检查与假 CLI，**未调用真实模型或训练库，没有 GPU 性能证据**。API 返回的 worker_reported_preflight 不等于独立验证的训练成功。

### 真实浏览器

实际使用系统 `/usr/bin/chromium`，Chromium **154.0.8037.57**，通过 `tests/browser_acceptance.cjs` 完成：

1. 示例加载/预览；50 对象任务上传；绑定任务的 train/calibration/validation/test/regression 上传。
2. 手动 0.8B 实验、不可变 Trial、CPU 作业提交；实际 Worker CLI 拉取/检查后页面显示“检查通过”，GPU 和 training_verified 仍为 false。
3. validation 外部预测导入与候选选择，再导入两侧 test 夹具预测；来源/不确定性提示和 report.json 下载。
4. **1440px 桌面与 390px 窄屏，四个视图无页面横向溢出；pageerror 为 0。** 修复真实发现的 Grid min-content 溢出、长 SHA/文件控件/状态文本问题。

命令：`KEV_BROWSER_OUTPUT=/workspace/scratch/kev-trainer-browser node tests/browser_acceptance.cjs`（使用环境已安装的 Playwright/系统 Chromium，不安装浏览器）。证据位于 `/workspace/scratch/kev-trainer-browser/`：manual-desktop.png、results-desktop.png、results-mobile.png、fixture-report.json。报告中的准确率只是已知夹具的手算断言，不是模型效果。

**更正早期记录：** 先前只检查 Playwright 默认缓存路径，浏览器下载遭 403 后误以为无可用 Chromium；补查发现系统 Chromium，已完成本轮真实浏览器验收。下面历史章节的浏览器阻塞已解除，未修改网络策略。

### 固定上游源码与真实阻塞

通过已连接 GitHub 只读工具读取固定提交的 train/benchmark/calibrate/checkpoint/data/api 和 calibrate_checkpoint 源码，逐文件验证 Git blob SHA，并记录 SHA-256/字节数及 AST 提取的 CLI flags 到 kev-source-contract.json。临时源码放在 /tmp，没有复制为本仓库的第三方实现。详见 KEV_CONTRACT.md。

源码确认 Noul criteria 是映射、dtype 只支持 bf16/fp32、LoRA 不支持 optimizer resume；checkpoint 不能只凭 adapter/head 文件存在就判加载成功。`kev.calibrate` 仅出报告，实际写 head.pt 的工具要求可信的训练/校准来源，custom JSONL 不能直接冒充已注册 suite。一步 smoke 不会输出 loss（每十步才记录），不能声称已满足 loss 检查。

尚未完成：GPU 多阶段执行器/共同时间预算、真实依赖和权重/tensor/targets、tokenizer、smoke 观测、上游兼容的校准来源登记、自动 benchmark/coverage、真实 loader/完整模型包；第二阶段自动 Claude trials/审查/用量和第三阶段 4B/多 Worker/部署。实际业务 50 对象目录也未随摘要提供。CPU Worker/源码检查/非模型 ZIP 均不能解除这些门禁。

### 工作区与发布

旧的 189 项补丁 kev_trainer-review.patch、190 项补丁 kev_trainer-review-current.patch 原样保留。本轮全部改动另存 kev_trainer-phase-one-review.patch，并在干净基线用 git apply --check 检查。没有提交/推送/PR、生产部署、初始化 Publish、持久账号凭据创建或秘密迁移。CI 文件是未提交草稿，没有远端 CI 结果。

下面保留历史证据；旧的“原设计/CLI 未知”“浏览器不可用”等结论由本节和当前需求清单替代。


## 云端开发与验收：2026-10-16 UTC

目标：`kirin7002/kev_trainer`，基线 `573b0d4ba3490c9d74d8ab08ff89297d346bff8c`。本次为 Linux、Python 3.12.14；从 requirements-lock.txt 新建临时 `.venv` 安装，未发布环境初始化配置。所有开发直接由 Codex 完成，没有调用 Claude Code 来开发。

### 基线与最终验证

- 基线：` .venv/bin/python -m pytest -q` 实测 151 passed；`pip check` 无依赖冲突。
- 最终：`.venv/bin/python -m pytest -q`：190 passed，0 failed，0 skipped；包括 39 项新增后端/进程/真实 HTTP 验收用例。
- `node --test tests/frontend.test.cjs`：5 passed，0 failed；对实际页面函数做乱序响应、重复刷新、取消选择、旧请求错误、特殊对象标签/ID 验证，并编译检查整页内联 JavaScript。
- `.venv/bin/python -m pip check`：No broken requirements found。
- `bash -n scripts/start.sh scripts/stop.sh`、`git diff --check`：通过。
- 既有非失败警告仍为 Starlette TestClient 对 httpx 的弃用提醒；未为消除警告擅自改变锁定依赖。

### 本轮真实进程 / HTTP 证据

`tests/test_http_acceptance.py` 在临时项目和数据目录、临时 loopback 端口，用实际 start.sh、server.py、FastAPI 应用和 stop.sh 完成：

1. 健康接口 200、中文首页 200；机器未接入、Claude 未安装状态真实返回。
2. 示例训练/测试数据导入，实验创建得到 train=8 / validation=2 / test=5。
3. 未安装 CLI 时生成方案返回 409；无方案训练返回 409，没有模型或训练任务。
4. 导入两份**人工夹具预测**，断言 baseline accuracy=0.2、candidate accuracy=1、delta=0.8、4 条改对；报告与下载一致，`source=user_imported_predictions`、`training_verified=false`、无计时时不可比较性能。这些数值仅用于测试正确性，不是实际模型效果。
5. 实际停止并核对 PID 记录清理，删除派生报告缓存，再启动实际应用；实验与预测保留，报告可按当前预测重建，随后再次正常停止。

其他新增进程测试使用临时假 CLI，验证：主进程先退出但子进程占有输出管道、关闭输出后继续运行、无界长行、畸形流事件、失败重试不返回旧方案。既有启动器测试继续验证重复启动和拒绝停止无关进程。

首轮之后继续复查，补充 `/train` 的状态检查/等待状态更新与方案操作共用互斥锁。新增并发 HTTP 回归证明慢连接检查不会把新接受任务的 `planning` 覆盖成 `waiting_for_machine`；在临时副本中仅移除这把锁，同一回归复现错误状态，修复后的工作区通过。测试仅使用离线机器桩与假 CLI，不连接 GPU 或远程机器。首轮 189 项的旧补丁保留，当前补丁单独导出。

### 修复与计算验证

- 数据/预测 JSON、嵌入 CSV 或字符串的 JSON 标签、实验参数中的无效 Unicode / NaN / 超大数 / 重复字段 / 深层嵌套不再产生未处理 500；非法上传不落盘。
- 可取消的输出读取限制单行 256,000 字符，总量 2,000,000 字符；队列和读取缓冲区有界，可靠传递结束标志，清理 CLI 私有进程组。
- 方案重试清除旧产物；详情与 JSON 下载在锁内读取快照。
- 报告 API 和 report.json 下载从当前预测重建，不信任可能陈旧/缺失的派生缓存。
- 大有限耗时均值不因求和溢出；逐对象指标改为累计计数。
- 单次 CPU 计算对照：5,000 条合成样本、7,500 个对象指标，与 HEAD 原算法完整结果相等；本环境原算法 5.8526 秒，修改后 0.0147 秒。这只是一次报告计算基准，不是 GPU、训练或推理性能，也不是稳定性能承诺。
- 补齐 GitHub Actions 工作流草稿；没有提交、推送、开 PR，因此**没有远端 CI 成功记录**。

### 尚未完成的真实验收

- 完整原始设计文档未在任务正文或仓库中提供；当前只对照仓库 README、AGENTS.md 与交接/验证文档，详见 DEVELOPMENT_STATUS.md、REQUIREMENTS_GAPS.md。父任务已另行安排只读调查，待原文到齐后补充设计对照。
- 本轮浏览器视觉验收未执行：Playwright 可用但 Chromium 未安装；向官方 CDN 下载返回 `403 Domain forbidden`。未修改网络安全策略，也未用既有本地截图冒充本轮云端结果。
- 真实 Claude 认证、CLI 兼容、联网、生成与实际费用没有验证；开发只用临时假 CLI。
- GPU / SSH / WSL2 适配器及训练执行器仍未实现接入；未连接机器、未启动训练、未下载模型、未测真实推理。完整训练闭环未完成。
- 没有迁移任何秘密、创建持久凭据、部署生产或发布安装/start 草稿。

## 迁移前历史记录（原文保留）

日期：2026-10-17。环境：macOS arm64、Python 3.12.14，独立 .venv。

## 已通过

- 全套 pytest：151 项通过，0 项失败，0 项预期失败。所有测试使用临时数据及假 CLI，不连接 GPU 或真实模型。
- 内联 JavaScript：Node --check 通过。
- 启动/停止脚本：bash -n 通过。
- 启动器真实进程测试：健康接口成功、重复启动被拒绝、仅停止所属进程、停止后清理 PID；异常记录指向无关进程时拒绝停止。
- 独立 API 验收：示例数据导入、8/2 训练验证留出、实验建立、两份外部预测导入、报告导出；手算总体/单对象/多对象准确率、F1 和 P95 与实现一致。
- 浏览器：示例导入、数据分布/样本预览、模型版本选择、实验创建和详情展示。修复了窄屏实验双列溢出，复验已上下堆叠。
- 凭据模式扫描：待提交代码未发现真实私钥、GitHub token 或 API key。运行目录、上传数据、认证文件、虚拟环境和日志均在 Git 忽略范围内。

## 验收发现并已修复

- 无效 UUID（包括 None）统一返回记录不存在，不再抛未处理 TypeError。
- 原子 JSON 写入失败清理临时文件。
- 大 stdin 发送纳入统一取消/超时控制，取消后清理读写线程和子进程。
- Noul questions 畸形内容返回 400；保留 instructions/criteria 和显式 false 标签。
- 耗时覆盖不完整时不标记速度可比。
- 方案调用禁用 hooks、CLAUDE.md、自动记忆、文件/命令工具和外部 MCP，仍保留用户认证。
- CSV 页面要求显式 [] 表示无对象，与后端缺失标签校验一致。
- 补充单/多对象 P/R/F1 和按对象数的报告展示。
- Uvicorn SIGTERM 后 PID 记录清理及无关进程保护。

## 尚未验证

- 真实 Claude Code 方案生成、账号认证、网络与实际费用。仅开发会话此前使用过 Claude Code；产品方案集成尚未做真实模型端到端验收。
- Linux 云端依赖安装、服务启动和测试：应由接续的 Codex 云端任务重新执行。
- GPU / CUDA / Windows WSL2 / SSH 机器适配器、训练、模型下载与实际推理性能，均留待后续联调。
- 公网部署、多人认证和生产持久化，不属于当前本机版。

已知非失败警告：当前 Starlette 对 httpx 测试客户端发出弃用提醒，现有测试全部通过；后续可单独评估 httpx2 迁移。
