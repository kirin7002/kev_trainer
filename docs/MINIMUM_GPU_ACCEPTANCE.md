# 第一阶段最小可执行验收材料

证据分级见 EVIDENCE_AND_BASELINE.md 与 UPSTREAM_EVIDENCE.json：依赖来自已下载核对的固定 uv.lock；base revision 来自固定 README 的声明；archive SHA 仅为发布资产 API 元数据，权重未下载核验。Hugging Face revision API 的独立请求受 403 阻断。

以下是运行者需要准备的具体材料，不要求传送任何秘密，也不授权本轮连接机器或训练。当前全部真实模型/CUDA结论仍未验收。可先完成离线业务数据审查，再在运行者明确批准的计算环境执行。

## 1. 一个计算环境和四个本地路径

- 单台 Linux 或 WSL2 Linux；Python 3.12/3.13；一张可被 PyTorch CUDA 使用的 NVIDIA GPU。提供非秘密的 GPU 型号、总/空闲显存、驱动版本、CUDA runtime 版本和 OS/Python 版本。显存下限目前没有实测结论，不能据 0.8B 名称承诺某张卡可训练；先执行有界 probe/smoke，OOM 必须失败并保留日志。
- 路径 A：干净的 jaredpalmer/kev Git checkout，HEAD **85ccf56a0afb952323c621055b02876e8498d60c**，包括 untracked 在内无改动。关键源码与 uv.lock 必须符合仓库 kev-source-contract.json 的 SHA。
- 路径 B：由这个 checkout 的 uv.lock 准备的独立 Python 解释器，例如 A/.venv/bin/python；不要安装 PyPI 同名 kev。runtime 当前核验的关键版本为 accelerate 1.15.0、datasets 5.0.1、numpy 2.5.3、peft 0.21.0、pydantic 2.13.5、scikit-learn 1.9.1、torch 2.8.0（CUDA build 可带本地后缀）、transformers 5.17.0。实际可安装性、CUDA build 和导入兼容尚未实测；若 lock 在目标环境无法解析，先报告准确错误，不擅自换版本。
- 路径 C：官方 release 的本地 `kev-0.8b.tar.gz`，45,813,161 字节，SHA-256 **0ae144c7675f0c3f333be0bb878a0f202ab9e6fa84169fb7cb16efe6176c1ef1**。release 来源见 kev-source-contract.json。
- 路径 D：独立可写 work root，训练与 evaluator 使用不同子目录；有足够空间保存每次 attempt 的原始 benchmark、日志、checkpoint 和 ZIP，不覆盖旧 attempt。控制端数据目录由单 FastAPI 进程独占。

## 2. 固定模型和离线缓存

- 平台模型 ID `kev-0.8b-v1.0`，父发布 `jaredpalmer/kev-0.8b@v1.0`；LoRA rank **16**、pointer head_dim **256**，不是可搜索参数。
- 基座 **Qwen/Qwen3.5-0.8B-Base**，revision **dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68**。目标训练解释器的 HF cache 必须已有该 revision 的完整 base 权重、配置、tokenizer/special tokens。仅有 adapter archive 不够。
- Worker 子进程强制 HF offline；不会自动下载或复制认证。真实 probe/compatibility 要检查 archive、head（weights_only）、adapter targets/tensors、base/revision、rank/head 与 dtype；smoke 还会通过真实 warm_start。当前 loader canary 需要真实 Checkpoint.load 和全任务目录 forward。
- 最终 ZIP 是完整 LoRA 契约：adapter/head/tokenizer/配置/校准/报告/日志/清单；**不内嵌基座，不包含 optimizer，不能作为 optimizer resume**。下载后需同一固定 base cache 的独立加载验收。已自动观察受支持的运行时镜像元数据，分别登记 config/manifest digest；无法取得时为 unknown/null，不能把环境变量或容器 ID 当成镜像证明。当前环境没有实际 digest；不要求提升权限或连接 daemon。见 EVIDENCE_AND_BASELINE.md。

## 3. 业务任务目录与数据

提供一份真实任务 JSON：`name`，1～100 项 `objects`；每项有唯一 `id`、`instructions`、`criteria`，criteria 为 true/false 描述映射。问题名称/含义由业务方给出，不能用 object-00 等合成测试目录代替。建议固定 `business_scope`，不同端分别建任务；`label_mode` 明确选择完整或逐条明确问题标注，见 [USER_GUIDE.md](USER_GUIDE.md)。

提供 train、calibration、validation、test 四个独立文件，regression 可选。每个文件绑定同一任务版本，保留原文件和 source_rule。逐条至少有稳定 id、非空 text/state；完整模式用完整审核的 intents 或全目录 Noul bool 标签，明确问题模式必须用非空 questions 中的显式 bool 标签，未询问项不参与训练和评测。`intents:[]` 只用于审核确认全对象均 false 的完整模式样本，不能代替未标注。具体判断遵循冻结业务问题与规则，不擅自猜测标签。

提供 conversation_id、来源/模板/近重复 group 等适用分组字段，避免一句话变体、同会话/同模板跨区。四区的文本和关联组必须互斥，旧反复调优用例放 regression，test 需未参与优化。业务方同时说明外部历史暴露；平台记录不能证明外部未看过 test。

接口层最小数据可很小，**业务验收不能据两条夹具作结论**。若需要当前组配对近似区间，最终 test 至少 30 个可信独立组且有可估计观测变化；逐对象正/负支持、多对象 2/3/4+、unknown、否定/疑问等覆盖与验收阈值由业务方审阅确认。样本不足只验证链路并标不确定。

## 4. 明确计算批准与验收顺序

运行者先选择学习率/epochs/batch/累积/seed/bf16 或 fp32、候选数量上限、实验统一分钟上限（包含基线、smoke、训练、校准、评测和上传），批准一次最小 smoke/候选验收。没有已验证的默认显存或收益保证。

API 仅在运行者显式启用 GPU 后工作；训练 Worker 与独立最终 evaluator 由各自进程继承不同的既有令牌。无需向聊天提供令牌值，不创建持久凭据，不修改网络策略。命令模板见 EXECUTION.md。

1. probe：真实 CUDA 计算、固定 checkout/lock/关键依赖/archive/base cache。
2. 准入与 smoke：严格 tokenizer 计数、无截断/丢弃、全对象训练分组、真实 warm_start/一步 optimizer、有限 loss/梯度/显存，无 OOM。
3. 可先运行独立 baseline-only 七阶段（不创建 Trial/不占候选数），取得 validation 基线后再进行一次完整候选：保存全部 13 阶段回执，零拒绝/截断，双方同等独立 calibration，validation 全 ID/逐条明确问题概率与 coverage，真实当前全目录 loader canary。
4. 运行者在另一受控解释器下载/核验模型 ZIP 和固定基座加载，实际取消/失租/重启验证进程已停、预算累计、失败不替换好模型。任何步骤失败应保留 attempt，不从 adapter/日志假装续训。
5. 只看 validation 选定/审查候选；再另行批准独立 evaluator 的一次最终 test。不得把 test 输出回传优化 Agent。最终收益、置信区间和业务阈值单独审阅。

本轮未执行以上真实步骤。第二阶段可离线导入/审批方案；真实 Claude 还需运行者指定准确 CLI `--version` 文本、可用 bare API 环境认证及明确额度批准，未授权时不调用，不必为第一阶段手动验收配置 Claude。
