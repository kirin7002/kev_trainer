# 第一阶段执行与验收边界

软件链路已实现并有独立合成契约回归。**本环境未安装 Torch/Kev、未下载模型、未连接 GPU、未执行真实训练。** 测试成功不等于模型效果验收；真实设备、权重和业务目录/数据仍需后续独立验收。

## 运行约束

平台继续使用单 FastAPI、中文 SPA、文件 JSON；只允许单服务进程、单执行槽。不能用多个 uvicorn workers 共享此数据目录。worker-queue.json 将作业归属、预算预留/扣减和终态存为一次原子替换；模型晋升单独有可恢复的 promotion_pending。不是多机事务队列。

CPU 准备默认可用。GPU 路径默认关闭；未来运行者需要在 API 进程显式设置 `KEV_ENABLE_GPU_EXECUTION=1`。API 与训练 Worker 从各自进程环境继承相同的至少 32 字符 `KEV_WORKER_TOKEN`；独立最终评测 Worker 使用**不同的** `KEV_EVALUATOR_TOKEN`。工具不生成或保存这些令牌，非本机 HTTP 被拒绝，重定向被拒绝。它们是受信 worker 的角色认证，不是硬件远程证明。当前页面仍按本机单用户设计，未扩展为公网多人服务。

未来显式运行 GPU Worker 所需材料：

- Linux/WSL2、Python 3.12/3.13、实际 CUDA；平台服务虚拟环境与训练解释器分离，禁止 CPU fallback。
- 独立、干净、包括未跟踪文件在内无改动的 `jaredpalmer/kev` checkout，HEAD 必须为 `85ccf56a0afb952323c621055b02876e8498d60c`。使用该 checkout 的 uv.lock 准备训练依赖；runtime 检查锁文件/关键依赖版本，不调用 pip 自动安装。
- `kev-source-contract.json` 中固定 release archive 的本地文件，SHA-256 `0ae144c7675f0c3f333be0bb878a0f202ab9e6fa84169fb7cb16efe6176c1ef1`；以及已按 models.json revision 缓存的基座/tokenizer。子进程强制 HF 离线，不下载数据、权重或读取进程认证令牌。
- 与平台相同代码指纹的 Worker；创建 Trial 后改代码需创建新 Trial，不能悄悄迁移旧配置。

未来运行命令模板（本轮没有执行）：

```bash
.venv/bin/python worker.py --role training --server http://127.0.0.1:8091 \
  --kev-source /operator/kev --kev-python /operator/kev/.venv/bin/python \
  --checkpoint-archive /operator/kev-0.8b.tar.gz --work-root /operator/training-work

.venv/bin/python worker.py --role evaluation --server http://127.0.0.1:8091 \
  --kev-source /operator/kev --kev-python /operator/kev/.venv/bin/python \
  --checkpoint-archive /operator/kev-0.8b.tar.gz --work-root /operator/evaluation-work
```

work root 只存 worker ID/服务地址/角色和 attempt 文件，不存认证令牌。注册先执行有界 CUDA/源码/依赖/archive probe；重启先按 PID 启动身份与 UID/私有 session 核验并停止旧进程，再刷新 runtime、领取新作业。PID 重用、启动窗口归属不确定、无法确认停止时保持占槽，不自动重派。恢复不承诺 LoRA optimizer resume。

## 真正串联的阶段

训练 Worker 仅收到 train/calibration/validation 和不含最终样本的配置：

`VALIDATING → BASELINE_RAW → BASELINE_CALIBRATION → BASELINE_CALIBRATING → BASELINE_SYSTEM → SMOKE → TRAINING → CANDIDATE_RAW → CANDIDATE_CALIBRATION → CALIBRATING → EVALUATING → LOADING → COMPARING`

- tokenizer 使用真实严格 encoder，记录每句 state、各问题 branch、packed token 数；不截断、不跳过。训练 50 对象按固定 5 问展开为 10 条，仍覆盖全 50 bool 标签；超过训练 context 的任何组均拒绝。评测保留全 50，只有 packed 总限额随实际计数增加，state/branch 上限不放宽。超长、OOM、CUDA/loader 不兼容均失败。
- smoke 一步实际 optimizer；用固定 `batch_loss` 的只读 scalar observer 记录真实 loss，原 tensor/梯度不变。正式训练拒绝静默丢记录，要求完整 epochs 覆盖、有限 loss/梯度和 optimizer step，记录峰值显存。进程成功退出本身不能晋升模型。
- 每次独立 attempt 子目录，每阶段一个私有进程 session；一个共享 monotonic deadline 包含取输入、baseline、smoke、训练、校准、评测、打包与上传。心跳只缩短预算；controller 按整个作业墙钟保守累计，失败/取消/过期也扣减，包含候选重试和最终评测。进程有独立 watchdog，取消/失租停止进程组，确认停止前不释放槽。不是 GPU utilization 或云厂商账单。
- 模型收益：同模板、T=1、阈值 .5 的 baseline/candidate validation；系统收益：双方都只用同一 calibration 分区拟合 T，再以 .5 评 validation。所有输出全 ID/50 对象概率、完整 coverage 和 raw rows 保留，controller 独立重算指标；低置信按任意对象 max(p,1-p)<.6 计数。首句冷耗时、余句暖耗时和串行模型吞吐排除加载/编码等开销，不是服务 QPS 或 SLA。

## 自定义 calibration 的明确适配

上游 `kev.calibrate` 只报告，而 `scripts/calibrate_checkpoint.py` 要求上游注册 suite，不能直接写任意 custom rows。本集成**没有冒充上游 CLI 原生支持 custom 数据**：使用固定 `kev.metrics.fit_temperature(**TEMPERATURE_FIT)` API（micro、121 点、.25～4），在平台先登记 train/calibration/validation 的文件 SHA、任务目录、分组与 raw rows SHA。逐行核对 calibration ID/全对象硬标签/clean variant/有限 raw logits/T=1/无 unknowable，再写工作区 checkpoint 副本的 Meta 与完整策略登记。

不传 `--allow-in-distribution`，不拟合 validation/test，不搜索阈值，不修改原父模型。基线获得相同校准机会。父模型预训练数据的历史重叠未知，策略会明确记录；平台组隔离不能证明外部历史无暴露。真实上游依赖、张量兼容及拟合/写入往返仍待 GPU 环境实测。

## 模型包与最终评测

ZIP 包包含 adapter/head/tokenizer、training config/metrics、固定 code/model/data 配置与源契约、两侧校准策略、逐文件长度/SHA、全部 benchmark 原始概率/rows/coverage、阶段日志和 loader canary 回执。拒绝路径越界、链接、重复/隐藏/加密条目、超限及哈希不符。controller 再核验阶段/数据/校准/coverage/loader 文件哈希后才把 active_model 指针切到新 attempt；失败不覆盖旧模型。

这是**完整 LoRA 产物契约，基座权重不内嵌**，下载文件明确 `base_weights_included=false/requires_pinned_cached_base=true/resume_supported=false`，不是离线自包含整基座，也不含 optimizer。只有真实 CUDA 来源且当前 `Checkpoint.load + DecisionModel.forward` 的 50 对象 canary 通过，用户模型下载端点才开放。实际下载可加载性尚未验收。

validation 人工选定后才冻结模型 ZIP SHA、validation 预测 SHA、两侧策略与候选；独立评测角色领取一次 test-only 作业：

`FINAL_VALIDATING → FINAL_RESTORING → FINAL_BASELINE_RAW → FINAL_CANDIDATE_RAW → FINAL_BASELINE_SYSTEM → FINAL_CANDIDATE_SYSTEM → FINAL_LOADING → FINAL_COMPARING`

只恢复已接受 T/阈值，不训练、不校准、不重新选择；controller 比对策略及 head/adapter 与冻结模型的 SHA。test 在领取前登记暴露，不能因失败、取消或改 ID 回流优化；排队尚未暴露的取消可解除冻结。训练角色无法冒用 evaluator ID 取得 test。最终报告包括模型/系统收益与全对象指标，不能将一次平台记录视为外部历史独立性的证明。

## 当前可审阅证据与剩余阻塞

`tests/test_gpu_execution.py` 使用显式 `software_contract_only` 测试后端、真实临时子进程和本机 HTTP，验证全链、失败、超时、取消、恢复、预算与完整性；仅测试依赖注入允许该后端，生产默认拒绝。所有合成产物/report 都是 `training_verified=false`，不能从真实模型下载端点取得。

真实 GPU/CUDA、固定依赖安装、缓存基座及 release 权重、tokenizer/LoRA/head 张量兼容、OOM/显存、真实 benchmark/校准写回/loader/取消恢复，以及用户真实 50 对象目录和合法分区，尚未验收。第二阶段自动方案审查/多候选/用量与第三阶段 4B/多 Worker/部署仍有独立缺口，见 REQUIREMENTS_GAPS.md。没有推送、PR、部署、凭据迁移或安全网络改动。

## 后续复查与第二阶段

已补进程组停止确认后才回 ACK、ps 查询有界、校准策略/回执一致、接受回执哈希和 ZIP 在完成/晋升重放中重查、pending 晋升占槽、时钟回拨保守扣减及外部 test 后冻结。第二阶段软件见 OPTIMIZATION.md；具体设备/模型/依赖/业务数据和验收批准清单见 MINIMUM_GPU_ACCEPTANCE.md。真实 GPU/模型结论仍未验收。

## 独立基线与运行时镜像

可先 POST /api/experiments/{id}/baseline，经 gpu_baseline_v1 七阶段获得 validation 基线；不创建 Trial、不训练、始终 training_verified=false，实际 CUDA 通过才可 baseline_verified=true。共享预算/租约，失败不换旧好基线。新方案固定引用已接受来源。镜像元数据会自动观察并明确区分 config/manifest digest，无法取得保持 unknown/null，不访问 daemon/socket/集群或云凭据。详细流程与证据边界见 EVIDENCE_AND_BASELINE.md。
