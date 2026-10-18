# Kev Trainer

完整功能清单与逐步操作见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)，包括业务隔离、1～100 个问题、标注模式、基线、训练准备、候选选择和最终验收。

独立的 Kev 微调平台：一个 Python 服务和一个中文网页。用户上传训练/测试数据、选择模型、生成微调方案，并比较评测结果。

当前已实现数据/方案准备、外部预测评测、手动 Trial、CPU 准备，以及默认关闭的 GPU 阶段执行、独立校准、benchmark、模型包晋升和 test-only 最终评测软件链路。本环境只做合成契约/临时进程验收，未连接 GPU、下载权重或训练真实模型。运行契约见 [docs/EXECUTION.md](docs/EXECUTION.md)，剩余验收见 [docs/REQUIREMENTS_GAPS.md](docs/REQUIREMENTS_GAPS.md)。

## 本地启动

要求 Python 3.12+，运行环境为 macOS、Linux 或 WSL2。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
./scripts/start.sh
```

打开 http://127.0.0.1:8091 。启动脚本保持前台运行，Ctrl+C 停止；也可另开终端执行：

```bash
./scripts/stop.sh
```

修改端口：`KEV_TRAINER_PORT=8093 ./scripts/start.sh`。单进程运行，启动器使用文件锁防止重复启动，停止时核对进程归属；不会停止其他项目。

所有上传内容与实验保存在 `.state/`；通过 `KEV_TRAINER_DATA=/absolute/path` 修改位置。运行记录在 `.run/`，两者均不进入 Git。

## 第一阶段：手动 0.8B 准备

1. 在数据页上传自己的任务目录 JSON：`name` 和 1～100 项 `objects`，每项包含 `id`、`instructions`、`criteria`。可固定 `business_scope`；`label_mode` 默认为 `complete`，也可选择 `explicit_questions`。Noul criteria 是描述对象，如 `{"true":"涉及该对象","false":"未涉及"}` 或 `{}`，不是字符串。平台不编造实际业务目录。
2. 导入绑定该任务版本的数据，保留原文件字节、来源规则、文件/内容 SHA。支持独立 `train/calibration/validation/test/regression`；也可先上传训练池，按会话/来源/模板/近重复关联组派生 train/calibration/validation。test 与 regression 不参与切分。比例按组近似，至少三个独立组，不保证标签分层均衡。
3. 创建手动 0.8B 实验，选择四个必需分区及可选 regression，设置候选数量和训练分钟上限。五类分区校验互斥。已有训练/测试两区的旧方案流程仍保留，但不构成新设计的完整训练流程。
4. 保存手动 Trial：自己填写学习率、epochs、batch、累积、seed 和 bf16/fp32。配置、数据/模型/代码指纹固定；不能搜索 rank、传任意命令、覆盖旧配置或把 adapter 当作 optimizer resume。达到候选上限会拒绝新增。
5. 可下载确定性的 `preparation.zip`，包含配置、源码契约、train/calibration/validation 的 System One JSONL 和文件哈希；按固定任务和每条明确问题转换，不含 test/regression。它明确是准备包，**不是模型**。
6. CPU Worker 拉取准备作业、验证哈希/分区/标签并回传结果。它不调用训练库。API 与 Worker 需从各自进程环境继承同一个至少 32 字符的 `KEV_WORKER_TOKEN`；平台不生成、写入或迁移账号凭据。未配置时只允许准备数据/配置，Worker 端点返回 503。已安全配置环境时，在另一个终端运行：

```bash
.venv/bin/python worker.py --server http://127.0.0.1:8091 --once
```

默认 API 仍仅本机访问；不改网络设置、不自动连接远程机器。取消/失租先停止进程组；过期必须确认旧进程停止才可重试。当前为单服务进程 JSON 队列和单执行槽，禁止多 uvicorn workers 共享目录。

手动比较先导入对应 Trial 的 **validation** 外部预测并选择候选，再到结果页做最终 test 配对；candidate JSON 额外带选定的 `trial_id`。完整 test 比较记录后不能再据它重选候选，替换预测的重复比较标为 regression。已在平台暴露的最终测试文本/关联组不能通过改 ID 再作为 train/calibration/validation/新 test。平台之外的历史使用仍未知。

原始概率可在每条预测中提供 `probabilities`（精确覆盖该样本明确问题的 0～1 数值；完整模式为全目录）。全样本覆盖才计算 binary Brier 和 10-bin ECE；逐对象列表覆盖整个目录，macro F1 只平均可定义项。报告包含分业务指标，未询问的问题不当作反例。配对收益按关联组计算大样本正态近似区间，少于 30 个独立组或零观测方差时标不确定。GPU 路径分别保存固定 T=1/阈值 .5 的模型收益和双方同等独立校准的系统收益；真实收益/性能尚无设备验收证据。

固定 Kev 源码、uv.lock、release archive SHA 已记录在 kev-source-contract.json，详见 [docs/KEV_CONTRACT.md](docs/KEV_CONTRACT.md)。未来启用 GPU 需要独立训练解释器、固定 checkout/离线权重，以及独立 evaluator 令牌；CPU 检查不会解除真实验收。完整 LoRA 包不内嵌基座，需缓存固定 revision，不支持 optimizer resume。不要通过 PyPI 安装同名 kev，那不是此模型项目。

## 已审查的多候选方案

第二阶段现在支持 v2 方案独立验证、准确 SHA 人工批准、预算内串行候选、validation 排名和停止条件；页面可离线导入，0 美元不调用 Claude。真实收费入口默认关闭，需固定 CLI 版本/bare 认证、运行者启用及本次额度批准；本轮仅假 CLI 验收。见 [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)。真实 GPU 最小材料和操作顺序见 [docs/MINIMUM_GPU_ACCEPTANCE.md](docs/MINIMUM_GPU_ACCEPTANCE.md)，不再笼统列为“待 GPU”。

独立原模型基线可在手动实验中先运行，且不消耗候选名额；镜像来源会自动观察或明确未知。固定版本、权重/锁文件证据的分级见 [docs/EVIDENCE_AND_BASELINE.md](docs/EVIDENCE_AND_BASELINE.md)，不把发布声明说成已加载权重。

## 旧版页面流程（保留兼容）

以下两分区/方案生成入口不是推荐的完整训练验收路线；新实验请按上面的手动 0.8B 流程及完整使用指南操作。

1. **数据集**：上传 UTF-8 JSON、JSONL 或 CSV，查看预览、标签分布及数据质量提示；也可加载明确标记的示例。
2. **创建实验**：选择训练/测试数据、Kev 0.8B 或 4B 固定版本，设置目标、说明、训练时长与方案预算。训练集按文本/分组留出验证数据，拒绝训练/测试重叠。
3. **实验与方案**：后台调用 Claude Code 生成结构化方案，展示日志，支持取消、重试和下载。关闭网页不影响后台任务；服务重启时未完成任务标记为中断。
4. **评测结果**：导入原模型与候选模型的外部预测，计算总体、单/多/无对象、按对象数及逐对象指标，查看错例改对/改错，导出报告。

GPU 未接入时“开始训练”会进入等待机器状态，不执行训练。外部预测报告明确标记 `source=user_imported_predictions`、`training_verified=false`，不代表平台完成了模型训练。

## 数据格式

JSON 数组，或 `{ "cases": [...] }`；JSONL 每行一个对象。标签必须明确给出，缺失标签不会当作 false。

```json
[
  {"id":"a","text":"打开车窗","intents":["window"]},
  {"id":"b","text":"空调和天窗都关掉","expected_result":{"intents":["air_conditioner","sunroof"]}},
  {"id":"c","text":"谢谢","intents":[]}
]
```

CSV 表头可用 `text,intents,id,group_id`，多标签用 `|` 分隔，无对象写 `[]`。`id` 可省略自动生成；`group_id` 可用于把同会话/模板的样本放到同一训练或验证分区。

也支持带明确布尔 `label` 的 Kev Noul JSONL，原始 questions（instructions、criteria、false 标签）会保留。第一版处理文本对象多标签任务。上传限制为每文件 10 MB、20,000 条样本。数据集为不可变版本，同一文本标签冲突与重复 ID 会拒绝。

## Claude Code 产品功能

开发工作由 Codex 直接完成，不再委派 Claude Code。**平台本身的“Claude Code 生成微调方案”仍是已确认的产品功能**；后续若需更换方案提供方，可另行调整。

安装官方 Claude Code 并在运行平台的同一账号下登录后，页面会调用本机 CLI。支持：

- `CLAUDE_BIN`：指定可执行文件，默认在 PATH / `~/.local/bin/claude` 查找。
- `CLAUDE_MODEL`：可选，指定方案模型；默认使用 CLI 的默认模型。
- `CLAUDE_TIMEOUT`：调用超时，默认 240 秒，可配置至 600 秒。
- `ANTHROPIC_API_KEY`：可选的 API 认证，不能写入代码或上传数据。

订阅登录保留 user 认证配置，方案子进程显式禁用 hooks、CLAUDE.md、自动记忆、文件/命令工具及外部 MCP。最终测试集不进入方案上下文。CLI 未安装、未登录、网络失败、超时或返回无效方案时会显示真实错误，没有伪造方案回退。

页面预算限制的是方案调用，CLI 的费用字段属于估算，不是实际账单。测试仅使用临时假 CLI，不消耗真实模型额度。

接受重新生成方案后，旧方案会被清除；新任务失败、取消或中断时不会把旧方案当成本次结果提供下载或训练。CLI 单条输出限制为 256,000 字符，总输出限制为 2,000,000 字符，取消和超时会清理本次 CLI 的进程组。

## 外部预测导入

在结果页复制当前实验的测试集 SHA。每份 JSON 必须包含全部测试 ID，不能只上传成功样本。以下只是格式示例，`dataset_sha256` 和记录需替换为自己的完整数据：

```json
{
  "dataset_sha256":"从结果页复制完整 SHA",
  "model":"your-model-version",
  "runtime":{"device":"实际 GPU 型号/机器标识","backend":"torch","dtype":"bf16"},
  "predictions":[
    {"id":"a","intents":["window"],"latency_ms":125.4}
  ]
}
```

`latency_ms` 可省略或为 null；提供时必须是非负有限数值。只有两侧运行信息相同且全部相同样本都有计时，才标记耗时可比较。指标包含集合准确率、micro P/R/F1、macro F1、逐对象指标、平均/P50/P95 延迟及计时覆盖率。无对象场景中分母为零的 P/R/F1 显示为不可定义，不伪造为 100%。

## 测试与云端接续

```bash
.venv/bin/python -m pytest -q
```

测试数据使用独立临时目录，不连接 GPU、远程服务器或真实模型；HTTP 验收仅启动临时本机服务。前端无构建依赖及外部 CDN。默认只允许本机访问，并校验请求 Host/Origin；公网部署需另加认证和访问控制。

JSON 输入拒绝重复字段、非有限数值、无效 Unicode 及超过 64 层的嵌套。报告与报告下载按当前两侧预测重新计算，磁盘报告缓存丢失或过期不会导致返回旧结果。

任务问题转换预计超过 128 MB 时会要求减小数据版本或精简任务说明。Unicode 表面归一化与显式组检查不等于语义近重复检测；字符上限不等于真实 tokenizer 验证，训练门禁会保留该未验证状态。

前端行为与整页 JavaScript 语法可用 Node.js 22+ 验证，无需安装 npm 依赖：

```bash
node --test tests/frontend.test.cjs
```

可选真实浏览器验收（环境已有 Playwright 和 Chromium 时；不调用模型）：

```bash
KEV_BROWSER_EXECUTABLE=/usr/bin/chromium node tests/browser_acceptance.cjs
```

`.github/workflows/ci.yml` 配置了锁定依赖安装、pytest、前端行为测试和启动脚本语法检查。该工作流需提交到 GitHub 后才能运行；本次工作区验证不等于远端 CI 已通过。

Codex 云端准备与接续任务见 [docs/CLOUD_HANDOFF.md](docs/CLOUD_HANDOFF.md)。本地 Claude 登录、SSH 私钥、数据、运行进程和虚拟环境均不会自动迁移到云端。
