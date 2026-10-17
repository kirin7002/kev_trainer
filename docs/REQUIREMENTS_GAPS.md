# 需求对照与验收

目标仅为 kirin7002/kev_trainer，基线 `573b0d4ba3490c9d74d8ab08ff89297d346bff8c`。原设计来源见 [ORIGINAL_DESIGN_HANDOFF.md](ORIGINAL_DESIGN_HANDOFF.md)。本轮继续实现第一阶段执行软件，没有重建仓库、丢弃旧改动或覆盖旧补丁。

## 第一阶段软件实现与证据

| 设计要求 | 已实现 | 真实验收边界 |
| --- | --- | --- |
| 50 对象独立 Noul | 显式版本目录、全 bool 标签、未知/缺失拒绝、固定模板与 true/false criteria | 真实车载目录未提供；object-00 等仅合成夹具 |
| 原文件与五分区 | 原字节/来源/SHA、转换版本、训练池按关联组派生三分区、会话/来源/模板/Unicode 表面近重复互斥 | 不宣称语义查重或外部历史无暴露 |
| 可复现 Trial | 手动固定 0.8B、冻结参数/base/revision/rank/head/code/data/seed、候选上限、独立 attempt | 不允许任意命令、rank 搜索或 adapter optimizer resume |
| 拉取/心跳/取消恢复 | 分离训练/evaluator 令牌，单槽原子 JSON 作业与预算账本；过期等待旧进程；PID 启动身份/UID/私有 session 核验；进程组停止和独立 watchdog | 真实临时进程与本机 HTTP 验证；真实 GPU 取消恢复未验收，不是多服务进程队列 |
| 训练阶段与预算 | 13 阶段真正串联，共享 deadline 覆盖输入/基线/smoke/训练/校准/评测/打包上传；失败/取消/过期也累计扣减；可恢复晋升 | 按作业墙钟保守计量，不是 GPU utilization 或云账单 |
| tokenizer 与 smoke | 严格 encoder、逐句 state/branch/packed 计数；训练按固定 5 问展开覆盖全 50；零丢弃/截断；一步实际 loss scalar observer、有限梯度/optimizer/正式全覆盖门禁 | 真实 tokenizer、张量、OOM/显存与 smoke 尚无设备实测 |
| calibration | 固定 fitting API、独立 calibration 文件/分组/全标签/raw logits SHA 登记；双方同等机会，阈值固定 .5；禁止 validation/test 拟合 | 显式平台适配，不冒称上游 custom CLI 原生支持；真实拟合/Meta 写回待实测 |
| benchmark 与指标 | 保存全对象概率/rows/coverage；固定 T=1 模型收益与同等校准系统收益；Brier/ECE/支持量/组不确定性；低置信/冷暖/串行吞吐/显存回执 | 少组/零方差标不确定；串行模型耗时不是服务 QPS；合成预测不是性能证据 |
| 模型契约与 loader | 当前 Checkpoint.load+forward canary、adapter/head/tokenizer/配置/策略/报告/日志/清单 ZIP；路径/链接/重复/超限/哈希拒绝；controller 再核验后晋升；失败保留旧指针 | 完整 LoRA 包不内嵌基座，需固定缓存 revision；真实下载包可加载性未验收 |
| validation 选择与最终 test | 选候选后冻结模型/validation/策略 SHA；独立 evaluator test-only 八阶段，无拟合/训练操作；训练角色不可冒领，领取前登记暴露 | 只追踪平台已知使用，不证明外部历史独立；失败后的已暴露 test 不能回流优化 |
| 中文页面和下载 | 准备、训练提交/阶段/取消、候选选择、冻结最终评测、报告与受限模型下载；真实 Chromium 1440/390px、JS 错误 0 | 默认 GPU 关闭；合成产物 training_verified=false，真实模型下载返回 409 |

详见 [EXECUTION.md](EXECUTION.md)、[KEV_CONTRACT.md](KEV_CONTRACT.md) 和 [VERIFICATION.md](VERIFICATION.md)。测试后端只通过隔离测试依赖注入启用，生产默认拒绝 software_contract_only，不会把假进程、假概率改名为真实 GPU 训练结果。

## 第一阶段真实阻塞

- 目标 Linux/WSL2 CUDA、固定 uv.lock 依赖、离线 base/tokenizer/release 权重和真实 adapter/head 兼容尚未验收。本环境未安装 Torch/Kev、未下载模型、未连接远程或启动训练；原生 Windows 不宣称已支持训练。
- 真实 tokenizer/smoke、校准拟合/写回、benchmark、完整模型加载、显存/OOM、实际 GPU 取消/掉线恢复和业务收益需要独立设备验收。已有源码契约不等于运行可用性结论。
- 实际完整 50 对象目录、合法分组和 train/calibration/validation/独立 test/regression 尚未提供。没有用合成 fixture 冒充业务数据。

## 第二、三阶段与架构差异

- 第二阶段软件已推进：v2 schema/白名单语义验证、六类安全输入、方案/输入双哈希快照、固定 CLI 版本及 bare 参数 probe、人工批准与单独开始、串行多候选、幂等候选/作业、validation 排名、目标/patience/基线退化/预算/失败停止、跨会话持久 Claude 保守预留。实际 CLI/账号/付费输出、服务商硬成本限制、真实 GPU 搜索收益尚未验收；已提供 baseline-only 独立作业，仍无已接受基线时输入明确 unavailable，资源须真实 smoke 准入。见 OPTIMIZATION.md，不宣称第二阶段真实产品验收完成。
- 第三阶段：4B 仅方案准备；多 Worker/任务、部署回滚、模型库、生产权限未验收。当前单槽 JSON 不作为多 Worker 生产调度。
- 原稿 Go/Vue/PostgreSQL/OSS 与已确认的 Python/中文 SPA/文件 JSON 有差异；本轮保留当前架构，没有擅自迁移。生产事务/对象存储、六页信息架构作为差异保留。

所有改动留在工作区。未提交、推送、开 PR、发布初始化配置或部署；旧三份补丁保留。没有迁移秘密、新建持久凭据、修改安全网络设置或访问 jev_demo。

## 本轮独立复查加固

- 进程停止后再次核验私有组；无法确认则不发送 process_stopped、不释放租约；进程身份/ps 查询有界。
- 接受回执和 ZIP 双哈希固定，完成/重启晋升重放重查；重复上传不能重签篡改缓存，pending 晋升继续占槽。
- 导出 calibration 策略与已验证报告一致，固定源码契约核验；数据/任务清单与冻结 Trial/model/limits 绑定，协调修改 run 清单也不能替换已冻结输入。
- controller 时钟回拨取消并保守扣完已分配预算；外部 test 预测导入后也禁止继续训练优化。注册刷新与领取由同一锁隔离，自动作业不能手动另开重试。
- 已知临时状态竞态已修复：生成结果在全部独立验证完成后一次转换到 WAITING_REVIEW，不短暂发布 PLANNING_FAILED。

第一阶段具体材料见 [MINIMUM_GPU_ACCEPTANCE.md](MINIMUM_GPU_ACCEPTANCE.md)。

## 本轮明确缺口补齐

- baseline_execution.py / baseline_commit.py 接入 gpu_baseline_v1 七阶段，实验级不创建 Trial，calibration/validation-only，无 optimizer，统一分钟与租约。新方案固定采用已接受基线，失败保留旧基线。
- runtime 自动观察受支持的 Podman 镜像元数据，区分 config/manifest digest；缺失明确 unknown/null，注册/报告/包清单一致性检查。当前环境真实 digest 未取得，不连接 daemon 或提升权限。
- 15 个选定上游文本文件与固定 commit tree/Blob/SHA 对照完成；archive 仅发布 API checksum、base revision 仅固定 README 声明、Hub tag 未独立解析，完整证据见 EVIDENCE_AND_BASELINE.md / UPSTREAM_EVIDENCE.json。依赖锁声明不代表实际安装可用。
- 固定上游 LoRA 不支持 optimizer resume；继续明确拒绝，不使用 adapter/日志构造半恢复。未推进 full_ft、多用户、增广/任意评分或第三阶段。
