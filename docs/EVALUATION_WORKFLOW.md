# 固定基线与 Kev 原生训练工作流

## 实验边界

不同业务默认分别创建实验、保存适配参数与决策头；可以从同一个发布版 Kev 继续微调，并复用相同的冻结 Qwen 底座。这里不涉及运行时自动切换适配器或共享 GPU，实际部署能力需后续验收。

新实验保存训练、验证和最终测试快照，以及 `protocol.json`。协议包含基线完整配置、Kev 源码版本、问题定义、选中阈值、集合指标版本、划分种子、各快照 SHA-256 和业务场景。问题定义只从训练资料和显式配置建立，不从最终测试集推断。

`business_scope` 用于单业务实验，拒绝混入其他场景。数据也可显式提供 `scenario`；场景会进入 Kev 的结构化 `state`，同时用于分场景报告，不只是报表元数据。未限定范围时可以显式混合场景，但不保证每个业务都改善。训练、验证、测试缺少某业务时展示覆盖警告；当前划分保持文本/分组隔离，并未保证按业务分层抽样。

标签数组是完整标注：未列出的已定义标签为否。Noul `questions` 每条只导出其明确标注的问题，不将未询问的标签补成否；允许各条样本问题数量不同。相同问题 ID 的规则须一致，不同含义应使用不同 ID。最终测试集的问题必须已由训练资料或用户配置定义。

## 顺序与接口

1. 创建实验，`GET /api/experiments/{id}` 返回协议及验证/测试阶段状态、数据指纹和基线匹配标识。
2. `GET /api/experiments/{id}/evaluations/validation` 导出无答案的固定推理输入。先由选定基线检查点推理，再导入验证基线。
3. 基线建立后才能请求训练。GPU/执行器未接入时返回真实离线状态，不执行命令。训练准备无需真实 Claude 调用，可选的 Claude 方案建议仍保留。
4. 候选使用相同验证输入、问题和阈值，导入后生成验证报告。可以更新验证候选，不能覆盖基线。
5. `POST /api/experiments/{id}/finalize` 锁定当前候选检查点和验证预测指纹，开放最终测试输入，并禁止本实验继续优化或替换验证候选。
6. 对保留测试集先导入基线，再导入已锁定候选。最终测试结果不能替换。重复提交相同文件可幂等接受。

预测导入为 `POST /api/experiments/{id}/predictions/{baseline|candidate}?split=validation` 或 `split=test`。文件格式：

```json
{
  "split": "validation",
  "protocol_sha256": "实验协议指纹",
  "dataset_sha256": "对应评测阶段的数据指纹",
  "model": "实际使用的精确检查点版本",
  "runtime": {"device": "实际机器标识", "backend": "kev", "dtype": "bf16"},
  "predictions": [{"id": "样本 ID", "intents": [], "latency_ms": null}]
}
```

基线 `model` 必须等于协议的 `baseline.checkpoint`。预测必须覆盖全部样本，不能跳过失败样本，不能预测未定义或该样本未询问的标签。阈值规则为概率大于等于阈值选中。外部文件的检查点、阈值及运行配置是导入者声明，平台未执行推理核验；报告记录 `training_verified=false`、`inference_configuration_verified=false`。

报告接口为 `GET /api/experiments/{id}/report?split=validation|test`，默认仍为 `test`。包含总体、单/多/无对象、逐标签、按业务场景指标及逐样本改对/改错。速度可比要求两侧完整计时且 runtime 相同。基线匹配标识只代表协议声明一致，不证明外部推理来源，也不自动跨实验搬用结果。

旧的无协议实验继续支持原来的最终测试导入；不会追认其验证基线、锁定状态或数据快照。使用新流程请新建实验。

## 原生训练准备与未完成项

从 `/api/experiments/{id}/native-training/` 下载 `training.jsonl`、`validation.jsonl`、`manifest.json`。清单含固定 Kev commit、基座 revision、`--init_from`、LoRA/head 配置和命令参数。默认起始参数为学习率 2e-5、2 轮、batch 1、梯度累积 8，不是已验证的最佳参数，也不会自动套用 Claude 建议。

文件应放在固定 Kev 源码检出下的独立实验工作目录，命令通过该 Kev 环境的 Python 执行，输出到该实验自己的 `checkpoint`。清单只声明命令，不安装、下载权重或执行训练；训练包不含最终测试文件。已核实固定版的 `--data` 路径不会再自动切出验证集，`--holdout` 是排除数据来源的参数；原生训练器会过滤超过 token 上限的样本，因此仍需在目标机器用真实 tokenizer 检查覆盖率、日志和显存需求。

GPU 接入后待完成：基线推理执行器、训练进程与超时/取消/恢复、检查点重载、候选推理，以及固定验证和最终测试的自动执行。参数更新、模型可加载和业务改善须分别取得真实证据；本轮接口和假 CLI 测试不替代 GPU 验收。
