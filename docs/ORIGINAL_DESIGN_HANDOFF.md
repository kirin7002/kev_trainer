# 原始初步设计的只读调查交接

2026-10-16，父任务从只读调查 `01a14647-14c2-73ae-ab6d-0bab42d630eb` 转交。来源：Mac `/Users/kirin/go_path/jev_demo/docs/kev-training-platform-design.md`，标题《Kev 训练平台：初步设计》，日期 2026-10-16。原文件未跟踪，未进入独立仓库；独立项目位于同级 `/Users/kirin/go_path/kev-trainer`。本文件记录已收到的范围摘要，不冒充原文件逐字副本，也不读取或修改 jev_demo。

## 阶段和验收

1. 第一阶段：50 个车载对象的独立 Noul 多标签任务；动作由下游执行。固定 Kev 0.8B 检查点、单 GPU、手动继续微调 LoRA + pointer head；数据导入/固定版本、基线候选比较、完整模型下载。验收从上传到可复现报告、取消真停、失败不覆盖模型、当前推理可加载模型包、test 隔离、收益能追溯到数据/参数/代码/校准。
2. 第二阶段：Claude 非交互 CLI、固定版本/bare、结构化方案、独立日志、人工审查，预算内多个候选；平台另行验证白名单、范围、兼容性与资源预算。候选数量、GPU 时长、Claude 金额由用户设置且需要计量。示例 lr=2e-5、2 epochs、3 trials 不是指定默认。
3. 第三阶段：4B、多 Worker、其他任务、部署/回滚。模型导出到推理服务，上线独立于训练平台。

## 必须保留的边界

- GPU Worker **拉取**作业并心跳，调用固定版本 kev.train / kev.benchmark / 校准工具；不是 SSH 推送。Linux/WSL2 优先，原生 Windows 另验。单 GPU 串行；租约过期必须先确认旧进程停止再重派。取消要杀掉真实进程。只有完整可读恢复状态才可续训，adapter/日志不算优化器检查点。
- 数据保留原文件，转换形成新版本；train / calibration / validation / test / regression 隔离。旧反复优化的 200 用例只能做 regression，另建 test。按会话、来源、模板、近重复 group 划分，检查未知对象、空文、重复、冲突、跨区近重复和 token 长度。未标注不得当 false；否定动作仍算涉及对象。训练可以拆问题组保留正例/困难/随机负例，评估/校准覆盖全 50 对象。增广标来源与审核，不改人工或 test 标签。
- test 标签及逐条结果不得进入优化 Agent、计划、训练、调参、阈值或候选排名。calibration 仅做校准；validation 选候选；test 一次配对验收，反复使用转 regression。置信区间按句/group，样本不足标不确定。
- 固定发布 checkpoint 与 base/revision 分记，init_from、rank、targets、head 兼容；rank 不是普通热启动搜索参数。记录父 checkpoint、base、Kev commit、镜像 digest、精度、模板/预处理版本哈希、温度阈值、数据版本、seed、参数和评估。
- Experiment 多 Trial，配置不可变、目录隔离、attempt ID，区分重试/续训/新实验。状态：QUEUED、VALIDATING、BASELINE、PLANNING、SMOKE、TRAINING、CALIBRATING、EVALUATING、COMPARING、COMPLETED，以及失败、取消、等待资源/用户。smoke 要看兼容、OOM、丢弃率、loss。保留清单、方案、代码 diff、训练日志、loss/梯度/显存/耗时、checkpoint、校准、逐样本 50 对象原始概率、指标/错误类型。预算、连续无改善、基础退化、达标均可停止。
- 评估包含集合准确率/单多/2、3、4+，micro/macro P/R/F1、逐对象支持量、改对/改错/漏误、Brier/ECE/低置信/unknown；avg/P50/P95、吞吐、峰值显存、冷暖分列和 GPU/Claude 用量。比较需相同样本/模板/目录/硬件/backend/精度/负载；区分固定模板阈值的模型收益与含模板校准的系统收益，基线同等校准机会。
- Claude 输入为 task_spec/data_profile/baseline_metrics/validation_errors/allowed_search_space/budget；输出 plan.json、plan.md、trial_configs，可选 preprocessing.patch/增广建议。方案要有 schema 版本、目标、假设、trials/参数/理由/停止条件；无 root、Docker socket、任意远程执行或生产 SSH 凭据。

## 初步架构与当前仓库的冲突

原设计写 Vue3/TS、Go API、PostgreSQL 队列/租约/预算、OSS 不可变数据与产物，及隔离的 Claude/GPU/评估器；六页与 Project/TaskSpecVersion/DatasetVersion/ModelVersion/Worker/Experiment/Trial/Evaluation/Artifact/AgentSession 对象。

当前已确认仓库要求一个 Python FastAPI 服务、中文单页、文件/JSON 存储。本轮不重写 Go/Vue、不擅自引入 PostgreSQL/OSS；先做单进程文件存储下的第一阶段数据/Trial 和拉取契约，明确它不提供数据库多进程事务或多 Worker 生产调度。现有 Claude 和 4B 选择属于提前实现的准备能力，不能当作第二/三阶段已验收。

固定训练库 CLI 源码现已只读核实（见 KEV_CONTRACT.md）；真实依赖、权重/加载器、实际 50 对象目录、真实 GPU 和 Claude 账号仍未配置，先实现明确的数据/配置/安全与进程契约，不发起真实训练。旧 Windows CUDA 文档只证明推理链路，不能证明训练。
