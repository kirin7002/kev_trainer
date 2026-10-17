# 第二阶段：可离线审查的方案与候选编排

当前已实现 v2 结构化方案的独立验证、准确 SHA 人工审批、预算内串行候选、validation 选择和停止条件。测试使用假 CLI 和明确的 software_contract_only 后端，未调用真实收费模型或 GPU。原有单方案原型保留，不能将它的旧 plan.json 当作 v2 已审批计划直接执行。

## 操作顺序

1. 在完整分区的手动 0.8B 实验新建自动方案，明确 Claude 实验总额度；0 美元可离线导入。实验总额度第一次确定后不能通过新会话重置或扩大。GPU 使用该实验原有统一分钟额度，候选数共享原有限额。
2. 获取六类输入，离线编写/导入 plan JSON，或在运行者显式启用后生成。平台独立执行 v2 JSON schema 和语义检查，不信任模型自己声称“通过”。
3. 审查人查看目标、假设、每个候选参数与理由、停止规则、原始输入/预算后，提交准确 plan SHA、姓名和审查意见。批准只创建不可变候选，不启动训练；拒绝可重新导入。当前本机单用户身份是自行填写审查人，不声称多人权限隔离或两个账号独立审查。
4. 单独点击开始。默认 GPU 关闭时停在 WAITING_RESOURCE，不生成假训练；启用后编排器一次只提交一个候选，后台每 0.5 秒查看结果。没有新数据库、外部调度器或多 GPU。
5. 每个候选真正执行第一阶段全链；只有完整晋升、回执校验且有 worker validation 的结果可排名。生产拒绝外部预测/测试夹具参与自动排名；测试依赖注入的夹具始终 training_verified=false。等分按方案先后，不能用 test 决胜。
6. 达标、连续无改善、超过允许基线退化、分钟预算耗尽、候选用尽、失败/取消均停止。失败不自动重试，不偷偷增加候选。best 仅从准确率不低于同协议基线的已接受候选中选择，min_improvement 用于判断有意义改善及 patience；这些规则是固定 validation_accuracy，暂不支持任意脚本或用户自定义评分。
7. 有合格候选时进入 AWAITING_FINAL_REVIEW，仅按 validation 选定，**不自动最终 test**。运行者需要另行审阅并提交第一阶段独立 evaluator。test 预测导入或最终评测冻结后不允许继续优化。

## 六类输入与不可变审查

接口 `GET /api/optimizations/{id}/inputs` 明确构造 task_spec、data_profile、baseline_metrics、validation_errors、allowed_search_space、budget，不复制整个 run 元数据或最终报告，不读取 test/regression 文件。

可在建立会话时提供本实验已晋升的 baseline_trial_id；只从其不可变 worker artifact 提取 **baseline validation** 指标和 validation 错例，固定 job ID，不根据后续候选/最终报告悄悄更新。没有该来源时，baseline_metrics/validation_errors 明确 available=false，附验证样本而非伪造错误。现已提供独立 baseline-only GPU 作业；在不创建 Trial 的情况下先测原模型。新建方案会固定采用最近已接受的基线，已有方案不会悄悄改写来源。仍无已接受基线时如实标 unavailable，审查者可拒绝资料不足的方案。

方案快照以 plan SHA、输入 SHA、来源组合标识；plan.json、plan.md、trial_configs.json、inputs.json 均保留。生成调用另保留六个独立输入 JSON 和 CLI profile。审查记录绑定 plan/input/proposal ID；开始前及每次新作业再次核验批准、候选身份/参数、代码和数据清单、外部 test 暴露。

只允许 learning_rate、epochs、batch_size、gradient_accumulation、seed、bf16/fp32。拒绝命令、任意路径、rank/base/head 更改、resume、预处理 patch/增广、test 指标、重复候选、超预算数量及 NaN。资源显存未知不会由 schema 验证伪称足够，实际 tokenizer/兼容/显存仍由真实 runtime/smoke 准入。

候选/作业 ID 从会话及方案候选名确定，重启/保存窗口重放使用同一身份；不会因此创建第二个作业或重置分钟。已审批自动候选不能经手动训练接口另开尝试。取消等待当前进程停止/租约确认后结束，不启动余下候选。PAUSED_ERROR 保持失败关闭，取消后查看原 attempt，不自动续训。adapter/日志不是 optimizer 检查点。

## Claude 付费入口的明确边界

默认关闭。未来运行者需要：

- 显式 KEV_ENABLE_CLAUDE_PLANNING=1；CLAUDE_BIN 指向可信 CLI；KEV_CLAUDE_EXPECTED_VERSION 为准确的 `--version` 文本。平台不会猜版本、安装或升级 CLI。
- 运行者已有的 bare API 环境认证；不会创建/保存/迁移凭据，不从聊天索取密钥。第二阶段不降级为 OAuth 非 bare。
- 每次请求明确 allow_paid_call=true 和 max_cost_usd，在本实验剩余额度内。版本/help 在有界私有进程中验证 bare、tools、schema、budget、MCP 和无会话参数，记录版本与二进制 SHA；工具/MCP/hooks/记忆关闭。CLI 不获得 GPU worker/evaluator/AWS/OpenAI 令牌。

调用前将**整个本次上限**保守记入持久额度，不因失败、超时、取消、重启、换会话或 CLI 回报较少而返还；CLI 回报费用另记，不能冒充账单。回执缺失/非有限/超过预留时拒绝计划，记录异常并阻止此实验后续收费调用。

**这是平台准入和保守额度账本，不是服务商账单硬封顶。** CLI 的 max-budget 在真实版本/账号上尚未验收，不能保证单次服务端请求不会越过设置值；严格财务硬限额需要运行者在服务商账号侧另行已有配置并验证，平台不创建或修改它。未提供认证/准确版本/明确额度时只可离线导入，本轮未做任何真实付费调用。

## API 与 v2 示例

- POST `/api/experiments/{id}/optimizations`：`{"claude_budget_usd":0}`；可选 baseline_trial_id。
- POST `/api/optimizations/{id}/import-plan`：`{"plan":{...}}`；GET 同一会话查看、GET `/plan.md` 下载。
- POST `/review`：plan_sha256、decision（approve/reject）、reviewer、notes；再 POST `/start`。
- POST `/advance` 可显式推进；后台会自动推进已开始会话。POST `/cancel` 取消。
- POST `/generate` 受上述收费门禁约束，运行测试仅使用假 CLI。

以下仅为格式示例，不是指定超参数或收益保证：

```json
{
  "schema_version": 2,
  "objective": "验证业务数据上的固定 0.8B 微调假设",
  "hypotheses": ["小学习率是否降低验证错误，需要真实实验检验"],
  "trials": [{
    "id": "candidate-a",
    "parameters": {"learning_rate": 0.00002, "epochs": 2, "batch_size": 1, "gradient_accumulation": 1, "seed": 42, "precision": "bf16"},
    "rationale": "仅改变白名单参数，先由真实 smoke 验证兼容和资源"
  }],
  "stop_conditions": {"metric": "validation_accuracy", "target_accuracy": 0.95, "min_improvement": 0.001, "patience": 2, "max_baseline_regression": 0.01}
}
```

## 尚未完成的真实验收与范围

假 CLI 测试验证参数、数据隔离、进程和费用回执处理；真实 CLI 固定版本的 bare/API/模型输出/取消/服务商成本限制、真实账号和 GPU 多候选收益尚未验收。当前不做模型生成后自动改数据、预处理/增广 patch、任意评分指标或多轮 test 驱动优化。第三阶段 4B、多 Worker/GPU、模型库和部署系统没有推进。

独立基线、镜像 digest 观察和源证据层级见 [EVIDENCE_AND_BASELINE.md](EVIDENCE_AND_BASELINE.md)。
