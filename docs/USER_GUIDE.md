# Kev Trainer 功能与使用指南

更新日期：2026-10-18。本文面向当前 main；业务场景与可变问题功能提交为 `902e591`。历史设计和交接记录中的“固定 50 问”不再是新任务的限制。

## 1. 项目是什么

Kev Trainer 是一个独立的 Kev 微调实验工作台：单个 FastAPI 服务、中文网页、文件/JSON 存储。它管理业务问题、数据版本、训练配置、Worker 作业和评测证据，不是通用云 GPU 租赁或生产推理服务。

推荐路线是：**一个业务建立一个任务与实验，从固定 Kev 父版本分别训练，使用同一份独立评测数据对比基线与候选。** 车端与 App 端规则不同时，不默认混合训练。复用基础权重不意味着共用业务适配参数；线上多适配器自动路由、热切换尚未实现。

这里的“问题目录”是业务判断定义，不是按问题数量创建独立物理决策头。任务可定义 1～100 个问题；这个软件上限不代表已验证真实设备能承受所有组合。

## 2. 已实现与验收边界

| 能力 | 当前实现与边界 |
| --- | --- |
| 业务任务 | 不可变目录、1～100 问、业务场景、完整/明确问题两种标注模式；任务哈希绑定配置 |
| 数据管理 | JSON/JSONL/CSV，原文件留存、内容 SHA、质量检查、标签统计、分组隔离、训练池派生分区 |
| 实验与 Trial | 固定 0.8B 手动训练配置、模型/代码/数据指纹、候选数量与墙钟预算；不修改既有 Trial |
| CPU 准备 | 下载确定性准备包，Worker 检查来源、哈希、标签和分区；不加载真实模型、不训练 |
| 独立基线 | 不占候选名额的原模型 validation 作业；GPU 默认关闭，真实基线仍待设备验收 |
| GPU 执行软件 | 源码/锁文件/权重检查、tokenizer、smoke、训练、校准、评测、加载、打包和晋升门禁 |
| 多候选优化 | v2 方案校验、准确 SHA 人工批准、单独开始、串行候选、validation 排名及停止条件 |
| 最终评测 | 先选择并冻结候选，再由独立评测角色运行 test-only；最终数据不提供给训练/优化 Agent |
| 报告 | 总体及单选/多选/无标签、逐对象、分业务指标、错例、概率校准指标、配对差异与计时覆盖 |
| 外部预测 | 可导入实际外部运行结果；报告明确不证明平台执行过训练 |
| 作业可靠性 | 单槽队列、租约/心跳、超时取消、进程组停止确认、重试和来源校验；不是多机调度平台 |

当前验证：349 项 Python 测试、6 项前端测试通过；合成数据的 7 问/业务隔离/明确问题模式完成桌面与手机 Chromium 流程验收。Python 有一项现有 Starlette/httpx 弃用警告。

**这些是软件验证，不是真实模型效果。** 本轮未安装真实 Kev/Torch 训练环境、下载模型、连接 GPU、训练业务模型、调用付费 Claude 或生产部署；合成产物均不能冒充已验证模型。Git 推送也不等于云环境 Publish、远端 CI 通过或新任务恢复验收。

## 3. 启动与检查

在仓库根目录、Python 3.12+ 环境执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
./scripts/start.sh
```

默认服务地址为 `http://127.0.0.1:8091`。可用 `curl --fail http://127.0.0.1:8091/api/health` 检查健康状态；健康正常不表示 GPU 已接入。启动脚本前台运行，Ctrl+C 或另一终端执行 `./scripts/stop.sh` 停止。

端口可用 `KEV_TRAINER_PORT` 修改，数据目录可用 `KEV_TRAINER_DATA` 指定绝对路径。默认 `.state/` 保存数据、实验与报告，`.run/` 保存启动记录；与 `.venv/` 一样不提交 Git。只能单服务进程使用此目录，不要配置多个 uvicorn workers。当前不是具备公网用户认证的多人服务。

## 4. 第一步：定义一个业务任务

进入“数据集”，创建不可变任务版本。以下仅是格式示例，不是生产业务规则：

```json
{
  "name": "车端控制示例",
  "business_scope": "car",
  "label_mode": "explicit_questions",
  "objects": [
    {"id": "window", "instructions": "是否请求操作车窗？", "criteria": {"true": "操作车窗", "false": "其他意图"}},
    {"id": "door", "instructions": "是否请求操作车门？", "criteria": {"true": "操作车门", "false": "其他意图"}}
  ]
}
```

问题 ID 需唯一，`criteria` 是对象，不是字符串。页面可指定业务场景及标注模式覆盖文件中的对应配置。保存后定义不可变；修改问题措辞、规则、数量、业务或模式，需要新建任务版本并绑定新数据。

`business_scope` 固定时，上传样本省略 `scenario` 会补为该业务，显式填写其他业务会拒绝。场景会进入模型实际输入，不只是报告备注。未固定业务的任务可接受逐条 `scenario`，但混合训练仍可能相互干扰，不保证所有业务同时提升。

## 5. 第二步：准备带标签的数据

两种模式不能混淆：

- `complete`（默认）：每条样本对整个目录都有确定答案。`intents` 是全部为真的问题列表，其他问题明确为假；`[]` 表示全否。若提供 Noul `questions`，必须包含目录全部问题。
- `explicit_questions`：每条样本只训练自己明确标注的问题，必须使用非空 Noul `questions` 和布尔 `label`。省略的问题既不是否，也不参与该条评测；不能用只有正标签的数组推断缺失答案。

与上面任务配套的 JSONL 单行示例：

```json
{"id":"car-001","scenario":"car","state":"打开车窗","questions":{"window":{"type":"noul","label":true},"door":{"type":"noul","label":false}}}
```

也可以只标注 `window`，但此时没有关于 `door` 的训练或评测证据。转换时使用冻结任务目录的 instructions/criteria，不能用样本中的临时说明悄悄改题。`state` 示例使用文本；平台会将场景与文本组成实际模型输入。

完整模式还可使用：

```json
[{"id":"car-001","text":"打开车窗","intents":["window"]}]
```

JSON 支持数组或 `{"cases":[...]}`；CSV 可用 `text,intents,id,group_id`，多个正标签用 `|` 分隔，全否写 `[]`。每个文件上限 10 MB、20,000 条，任务目录上传上限 64 KB。数据保留原文件和规范化版本，拒绝重复 ID、同场景同文本冲突、无效布尔标签、未知问题等错误。不同场景相同文本可有不同答案，但这不会取消跨分区近重复/关联组检查。

## 6. 第三步：分开五类数据

| 分区 | 用途 | 禁止事项 |
| --- | --- | --- |
| train | 更新参数 | 不混入最终测试或其关联组 |
| calibration | 独立拟合温度校准 | 不用于训练；不替换为 validation/test |
| validation | 比较候选、选择配置 | 不把它当最终独立效果证据 |
| test | 候选锁定后的最终验收 | 不参与调参、校准、方案生成 |
| regression（可选） | 已暴露样本的持续回归检查 | 不宣称新的独立测试收益 |

上传时选择角色并绑定同一任务版本。也可上传训练池，再按关联组派生 train/calibration/validation；至少三个独立组，比例按组近似，不保证类别均衡。最终 test 和 regression 要另行准备，不参与派生。

用 `group_id` 或来源/会话/模板信息标识关联数据。平台进行显式组和 Unicode 表面近重复检查，但不能证明语义无泄漏，也不知道父模型或平台外历史是否看过数据。已暴露的最终测试文本/关联组不能通过改 ID 回流训练或当作新 test。

## 7. 第四步：创建实验并先做基线

在“创建实验”使用手动 0.8B 流程，选择任务、train/calibration/validation/test 和可选 regression，填写候选数及训练分钟上限。检查任务场景、标注模式、各分区 SHA 和业务覆盖提示。

具备已批准并验证的 GPU Worker 后，先点击“只测原模型基线（不创建 Trial）”。它不训练、不占候选名额，但计入统一墙钟预算。未接入设备时不要用模拟数值填充基线。

对比必须锁定同一模型父版本、问题规则、数据版本和评测协议。换了评测数据，就需要该数据对应的新基线；不得拿旧数据上的分数与新候选比较。平台训练链也会按冻结输入执行两侧基线/候选评测；缓存或引用只可用于来源和配置一致的结果，不能只看模型名字。

## 8. 第五步：保存 Trial 并完成 CPU 准备

填写学习率、epochs、batch、梯度累积、seed 和 bf16/fp32，点击“保存 Trial”。每个 Trial 固定配置、代码及模型/数据指纹；改代码或参数后创建新 Trial，不覆盖旧记录。当前不支持任意命令、自由搜索 LoRA rank 或 optimizer resume。

下载 `preparation.zip`，核对配置、源码契约及 train/calibration/validation JSONL 和哈希。**准备包不是模型，且不包含 test/regression。** 训练问题按每组最多 5 问展开，数量取决于每条样本实际标注，不把未询问的问题补成负例。

若 API 与 Worker 已由运行者安全配置相同的至少 32 字符 `KEV_WORKER_TOKEN`，可运行：

```bash
.venv/bin/python worker.py --server http://127.0.0.1:8091 --once
```

然后在页面提交 CPU 准备检查、查看作业状态。未配置认证时 Worker 端点返回 503，但仍可准备和下载文件。不要把令牌贴进聊天、命令示例或仓库。CPU 通过只说明准备契约通过，不说明 tokenizer、GPU 或真实模型可运行。

## 9. 第六步：真实训练，或导入外部 validation

真实训练需运行者另外准备固定 Kev 源码、独立锁定依赖的解释器、本地 archive/缓存基座及 CUDA，明确启用执行并启动训练 Worker。材料与命令见 [MINIMUM_GPU_ACCEPTANCE.md](MINIMUM_GPU_ACCEPTANCE.md) 和 [EXECUTION.md](EXECUTION.md)。不要通过 PyPI 安装同名 `kev`，也不要在未批准时下载权重或启用付费服务。

执行软件依次验证输入、原模型、校准、smoke、训练、候选评测、重载和模型包；OOM、超长、丢记录、非有限 loss、来源/哈希/覆盖不符都会失败。取消或失租需确认旧进程停止后才释放槽；重试是新 attempt，不承诺续接 optimizer。

没有 GPU 时，可以完成数据/配置准备，或导入真实外部运行的 validation 预测。预测要绑定该分区 SHA、覆盖全部 ID，并选择对应 Trial。示例结构如下，SHA 与 ID 必须替换为自己的完整分区：

```json
{
  "dataset_sha256": "当前分区完整SHA",
  "model": "实际模型版本",
  "predictions": [
    {"id":"car-001","intents":["window"],"probabilities":{"window":0.9,"door":0.1},"latency_ms":12.3}
  ]
}
```

`probabilities` 可选；提供时必须精确覆盖该样本明确标注的问题，不能缺项或增加未询问项。完整模式即覆盖全目录。标签与概率应由同一实际推理协议产生。外部结果始终标记 `source=user_imported_predictions`、`training_verified=false`，不能证明平台训练或校验了模型。

多候选方案是可选替代入口：先离线导入/生成 v2 方案，独立校验后按准确 SHA 人工批准，再单独开始；系统按 validation 排名并执行停止条件。付费 Claude 默认关闭，真实提供方费用硬限额未验收。详细操作见 [OPTIMIZATION.md](OPTIMIZATION.md)，不要把旧版方案生成按钮当作新执行链的必要步骤。

## 10. 第七步：选择候选，再做最终 test

先看 validation 的总体和分业务结果，确认哪些错例改善、哪些退化，再选择 Trial。明确问题模式只统计已标注范围，不能把少问问题带来的高分当作全目录识别提升。每个业务分别验收，不只看混合总分。

GPU 路线点击“冻结候选并提交一次最终 GPU 评测”，由独立 evaluation Worker 使用不同的 `KEV_EVALUATOR_TOKEN` 执行 test-only。它恢复冻结模型与校准策略，不训练、不重新校准、不根据 test 重选候选。训练角色不能领取最终测试。

外部路线在结果页导入 baseline/candidate 的同一 test 分区预测，候选文件额外提供 `trial_id`，值必须等于选定 Trial。首次完整比较后不再据此选优；替换预测的重复比较标为 regression。两条路线都不能把失败、取消或改 ID 当作消除已暴露历史的方法。

## 11. 第八步：读报告、保存证据

先看来源、`training_verified` 和数据/模型 SHA，再看集合准确率、micro P/R/F1、macro F1、逐问题和分业务表现、漏选/误选及改对/改错样本。分母为零的指标显示不可定义，不当作 100%。仅有部分标注的报告不是全目录完整正确率。

全样本原始概率覆盖后才计算 Brier/ECE。当前阈值固定 0.5，不提供已验证的阈值搜索；模型收益用双方 T=1，系统收益给双方同等的独立 calibration 温度拟合机会，不用 test 校准。

耗时只有两侧运行信息一致、全部同样本均有计时时才可比；训练 benchmark 不是线上 QPS/SLA。配对收益按关联组给近似区间，少于 30 个独立组或零观测方差标为不确定。业务通过条件应在看最终 test 前约定，程序跑完不代表效果达标。

可以下载报告；真实模型下载还要求 CUDA 来源、产物完整性与原生 loader canary 门禁通过。模型包包含 LoRA/head/tokenizer/策略和证据，但不内嵌基础权重，不是自包含整模型，也不能作为 optimizer resume 文件。合成夹具不能下载为真实模型。

## 12. 下一步与维护规则

下一步是先确认一个业务的正式问题目录与合法分区，再按材料清单申请真实 GPU 小规模验收：**基线 → smoke/训练 → 保存重载 → 同 validation 对比 → 锁定 → 同 test 最终对比**。随后才扩大数据、候选和其他业务。真实效果、显存/速度、训练依赖/张量兼容、线上适配器路由、多 Worker、4B 完整训练链和生产部署均不能据本轮测试宣布完成。

开发前先检查工作区、fetch/pull 最新远端 main，阅读新增提交与交接，再确定范围。不能覆盖未提交工作或强推。检查命令：

```bash
.venv/bin/python -m pytest -q
node tests/frontend.test.cjs
bash -n scripts/start.sh scripts/stop.sh
```

其他文档：[运行契约](EXECUTION.md)、[优化流程](OPTIMIZATION.md)、[基线与来源证据](EVIDENCE_AND_BASELINE.md)、[真实 GPU 最小验收](MINIMUM_GPU_ACCEPTANCE.md)、[云端交接](CLOUD_HANDOFF.md)。本指南描述当前用户流程；历史记录保留用于追溯，不将旧的未提交状态或固定 50 问示例视为当前限制。
