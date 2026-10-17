# 固定版本证据、baseline-only 与镜像来源

2026-10-16。本轮只读取公开源代码/发布元数据，不连接 GPU、不启动真实训练、不下载模型权重或迁移认证。机器可读记录见 [UPSTREAM_EVIDENCE.json](UPSTREAM_EVIDENCE.json)。以下明确区分下载字节校验、发布者声明和真实运行验收。

## 实际读取的来源和证明范围

| 项目 | 实际读取来源 | 本轮核对 | 尚不能证明 |
| --- | --- | --- | --- |
| Kev 固定 commit | [GitHub commit API](https://api.github.com/repos/jaredpalmer/kev/commits/85ccf56a0afb952323c621055b02876e8498d60c)、其 tree `818dcca022d4c03ccc03f955131dcd8a0d99c763` | API 返回 commit `85ccf56a0afb952323c621055b02876e8498d60c`；15 个选定文本文件按 tree Git blob/SHA-256 核对 | 没有克隆/构建/运行整库，不把源码存在当作模型可用性 |
| CLI/loader 接口 | 固定 commit 的 `kev/train.py`、`benchmark.py`、`checkpoint.py`、`model.py`、`metrics.py`、`predictors.py`、`suite.py`、`api.py`、`data.py`、`calibrate.py`、`scripts/calibrate_checkpoint.py` | 已下载选定文本并核对 12 项 runtime 源文件/lock 的 hash；具体路径/字节数/Blob/SHA 在 JSON 中 | 没有安装/导入真实 Torch/Kev 或执行其模型函数 |
| 训练依赖 | [固定 uv.lock](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/uv.lock) | 335,387 字节，Git blob `e4dacc257320a1c891ad481fc0dde6fcb2af51c7`，SHA-256 `a9922dbb89acdef78299fd2b4a8c3f7f0fa1b2bc08b55595b6926fa785a9c466`；TOML 解析核对八项版本 | 这是 lock 声明，不是本环境已安装/目标 CUDA ABI 可用的证明；平台 requirements-lock.txt 与训练 uv.lock 是两套环境 |
| 完整 base revision | [固定 README 第 335 行](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/README.md#L335) | 下载 README，Git blob `db45dd5e4e9c5d6f51b3be9d9605f92dd1cf9b77`；正文训练示例明确 `Qwen/Qwen3.5-0.8B-Base` 和 `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | **发布者源码声明**，未下载该 revision 的 config/tokenizer/权重，不能证明本地缓存或张量真实性 |
| 父 checkpoint 别名/结构 | [固定模型卡](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/docs/model-cards/kev-0.8b.md) | 24,767 字节，Git blob `fe70d102ac32ad5867059d932f6314696aa8ac80`；卡中说明 0.8B 基座、rank/head，以及 `@v1.0` 发布用法 | 模型卡是发布者声明，尚未独立解析 Hub 的 v1.0 tag 到完整 commit 或加载 head/adapter |
| archive SHA/大小 | [GitHub release API](https://api.github.com/repos/jaredpalmer/kev/releases/tags/kev-1.0) 的 asset `603844150` | 文件名 `kev-0.8b.tar.gz`，45,813,161 字节，API digest `sha256:0ae144c7675f0c3f333be0bb878a0f202ab9e6fa84169fb7cb16efe6176c1ef1`；[下载地址](https://github.com/jaredpalmer/kev/releases/download/kev-1.0/kev-0.8b.tar.gz) 仅记录 | **仅发布资产元数据**。本轮没有下载 archive、解包或独立计算 archive SHA。真实 worker 会对运行者提供的本地 archive 实际求 hash，不符即拒绝 |
| 上游 release 验证报告 | 固定 `runs/release/kev-1.0-verify/tar-0.8b/report.json` | 下载该公开报告，Git blob `a0f7b816cb05ef34060289d0da48d2155ad400e5` | 这是上游报告，不是本平台复现实验，也不代替业务验收 |

锁文件实际解析结果为 accelerate 1.15.0、datasets 5.0.1、numpy 2.5.3、peft 0.21.0、pydantic 2.13.5、scikit-learn 1.9.1、torch 2.8.0、transformers 5.17.0。没有按名称推测接口。模型卡中的推理显存描述也不作为本平台训练显存保证。

Hugging Face 以下 API 的独立核对失败：`/api/models/Qwen/Qwen3.5-0.8B-Base/revision/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` 和 `/api/models/jaredpalmer/kev-0.8b/revision/v1.0`；本环境直接请求为 tunnel 403，web open 亦不可用。没有绕过网络限制，也没有用第三方衍生模型的声明填补这个证据缺口。

## baseline-only 的实际软件链路

新增 `POST /api/experiments/{id}/baseline`、`GET` 同路径及页面入口。GPU 未显式启用返回 409，不创建作业。启用后由训练角色的 `gpu_baseline_v1` 能力拉取，七阶段：

`BASELINE_VALIDATING → BASELINE_RAW → BASELINE_CALIBRATION → BASELINE_CALIBRATING → BASELINE_SYSTEM → BASELINE_LOADING → BASELINE_COMPARING`

这是实验级作业，trial_id=null、不创建 Trial、不占候选名额；config 没有训练参数，只接收 calibration/validation，排除 train/test/regression。原模型在 T=1 与独立 calibration 拟合 T 两种协议下测 validation，保留原始全 50 概率/rows/coverage、校准/loader/来源/日志。没有 optimizer step，没有训练或候选恢复操作；始终 training_verified=false。真实执行通过才可 baseline_verified=true，合成后端两者均 false。

使用原有单槽、租约/fencing、统一分钟预算和停止确认，过期/失败也扣减；接受后经 controller 独立校验并发布 baseline_evaluation 指针，失败保留旧好基线，不替换已训练模型。baseline ZIP 是评测证据包，包含核验所需的父 adapter/head 副本，不是新训练模型下载或 optimizer 检查点。

之后新建第二阶段会话会固定引用最近已接受的 baseline job，将其 validation 指标/错例送入六输入；已有会话不会因新基线完成而偷偷改写已审查输入。需要新来源时新建/重新审查方案，不能把最终 test 报告拿来补基线。

## 镜像 digest 自动观察与未知值

runtime probe 自动读取已存在的、受限大小/归属/权限的 `/run/.containerenv`。依据 [Podman 官方文档](https://docs.podman.io/en/v6.0.2/markdown/podman-run.1.html)，某些运行模式才提供 engine/image/imageid；文件缺失或不含信息是正常情况。**本平台不要求启用 privileged，不连接 Docker socket，不调用集群/云元数据 API、不改变权限或凭据。**

- 从受支持的 runtime 元数据读取 imageid，只登记 **config_digest**；绝不把它冒充 OCI manifest digest。
- runtime 的 image 引用本身是 `repo@sha256:...` 时另登记 manifest_digest。记录来源和元数据 SHA，不声称密码学独立认证（independently_attested=false）。
- 缺失、不支持的 engine、可写/链接/超大/坏值，均返回 status=unknown、两个 digest=null。环境变量、container ID、mount 路径 hash 都不能填充证明。
- 注册、作业 runtime、报告和包清单记录该来源；controller 校验一致性，变化时拒绝晋升。创建 Trial 时尚不知道 worker 的镜像，原 config.image_digest=null 保持真实，不事后篡改冻结配置。

当前开发环境实际调用该纯 stdlib 采集函数，结果为 **unknown/null**；没有可用实际镜像 manifest digest。Docker/Kubernetes 等没有受支持元数据时也保持未知，本轮没有伪造“自动获取成功”。测试覆盖的是合成元数据解析/拒绝和回执一致性。

## 恢复契约：明确拒绝 LoRA optimizer resume

[固定 train.py 第 472–473 行](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/kev/train.py#L472) 对 resume/save/stop 相关状态明确要求 full_ft。第一阶段是 LoRA，因此不能启用该 full-weight 恢复路径。没有核实/实现 LoRA optimizer/scheduler/RNG/数据游标完整状态，平台拒绝 resume_from，并且不生成 --resume。

Worker 重启“恢复”只核验/停止旧私有进程与确认租约，不恢复优化器；重试是新 attempt，重新从固定父 checkpoint 开始，保留旧目录并累计预算。新方案是新 Trial；adapter、日志或推理模型包都不能当作半恢复检查点。本轮未扩展 full_ft 或第三阶段。
