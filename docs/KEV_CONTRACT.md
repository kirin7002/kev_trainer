# 固定 Kev 源码契约

本轮通过已连接的 GitHub 只读工具取得 `jaredpalmer/kev` 的固定提交 `85ccf56a0afb952323c621055b02876e8498d60c`。没有安装 PyPI 的同名 `kev`（上游明确提示那是另一项目）、没有安装 torch/训练依赖、下载权重或执行训练。浏览器直取 GitHub 失败不等于源码不可取得；连接器的固定 ref 读取成功。临时源码逐文件校验 Git blob SHA，SHA-256 与字节数记录在 `kev-source-contract.json`，不将上游完整源码复制进本仓库。

已核实的接口：

- [kev.train](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/kev/train.py)：`--data/--base/--base_revision/--init_from/--lora/--lora_targets/--head_dim/--lr/--epochs/--batch/--accum/--dtype/--device/--checkpointing/--seed/--out/--max_steps`。dtype 仅 bf16/fp32，bf16 要求 cuda；拒绝覆盖既有 out。LoRA 路径不支持 `--resume`，上游只给 full_ft 保存 optimizer resume；平台不能把 adapter/日志当作续训检查点。
- [kev.api](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/kev/api.py)：Noul criteria 是 true/false 描述映射，而非字符串。转换据此固定；[kev.data](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/kev/data.py) 用 `_meta.id/group_id/source/split` 保留稳定身份和分组。
- [kev.benchmark](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/kev/benchmark.py)：本地 `--run`、`--data`、`--out`、`--device`；输出 predictions.jsonl、rows.json、report.json；custom data 的过长样本可能被拒绝并计入 coverage，因此不能只看进程退出码或成功样本。
- [kev.calibrate](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/kev/calibrate.py) 只生成报告；真正写入 head.pt 的 [calibrate_checkpoint.py](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/scripts/calibrate_checkpoint.py) 会验证拟合行的 suite/partition/训练来源。普通 custom JSONL 不能自动证明满足其来源检查；平台不使用 `--allow-in-distribution` 绕过、不在 test/validation 上拟合、不手工填一个温度冒充校准。
- [kev.checkpoint](https://github.com/jaredpalmer/kev/blob/85ccf56a0afb952323c621055b02876e8498d60c/kev/checkpoint.py)：LoRA adapter + head.pt + tokenizer，加载时还需要 pinned base；warm_start 验证 base/revision/rank/head/option_isolation/special_embeddings/weights，再验证 adapter tensors。仅文件存在和哈希一致不能证明可加载。

`kev_contract.py` 保留纯源码/argv/结构审阅功能；实际执行由 `trial_execution.py`、`final_evaluation.py`、隔离的 `kev_runtime.py` 和 `pipeline_commit.py` 实现。GPU 默认关闭，源码 checkout/uv.lock/关键依赖/官方 archive SHA/平台代码指纹必须一致。测试只运行明确的合成后端，本环境没有 Torch 或真实训练。

已新增固定 `model.py/metrics.py/predictors.py/suite.py/uv.lock` 的 Git blob/SHA，官方 release archive 的 GitHub API digest/字节数及关键依赖版本登记。没有复制上游全库或下载权重。custom calibration 使用固定 fitting API 加平台独立分区/原始行注册，明确区别于上游 suite-only 写回 CLI；不绕过来源校验，不拟合 validation/test。完整适配、13+8 阶段、预算、模型包与实际 loader 门禁见 [EXECUTION.md](EXECUTION.md)。

尚需真实运行环境、缓存权重、CUDA、tokenizer、targets/tensors、校准写回和 loader 验收。源码契约与合成测试不构成 runtime/training_verified=true 的证据。

## 后续证据复核

已重新读取固定 commit API/tree、uv.lock、README、模型卡、train.py 和 release asset 元数据。15 个选定文本按 Git blob/SHA 对照；完整 base revision 是 README 的静态声明，archive SHA 是发布 API digest，尚未下载 archive/权重；Hub revision API 独立核对受 403 阻断。详见 EVIDENCE_AND_BASELINE.md / UPSTREAM_EVIDENCE.json。固定 train.py 472–473 明确 LoRA 不支持 optimizer resume，本平台保持拒绝。
