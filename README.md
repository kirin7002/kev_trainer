# Kev Trainer

独立的 Kev 微调平台：一个 Python 服务和一个中文网页。用户上传训练/测试数据、选择模型、生成微调方案，并比较评测结果。

当前已实现非 GPU 部分。GPU Linux / Windows WSL2 适配器尚未接入，不会产生模拟训练进度、模型文件或准确率。

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

## 页面流程

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

测试数据使用独立临时目录，不连接 GPU、服务器或真实模型。前端无构建依赖及外部 CDN。默认只允许本机访问，并校验请求 Host/Origin；公网部署需另加认证和访问控制。

Codex 云端准备与接续任务见 [docs/CLOUD_HANDOFF.md](docs/CLOUD_HANDOFF.md)。本地 Claude 登录、SSH 私钥、数据、运行进程和虚拟环境均不会自动迁移到云端。

新实验的固定基线、验证与最终测试顺序，以及 Kev 原生训练文件导出见 [docs/EVALUATION_WORKFLOW.md](docs/EVALUATION_WORKFLOW.md)。不同业务可分别建实验，复用相同基础权重并各自保存适配参数；GPU 推理和训练执行器仍待接入，导出命令不代表已训练。
