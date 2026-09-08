# infra-probes/B14 · verl MoE 真实训练画像

状态：**Q21 场景冻结完成；Q22 GPU 画像未开始。** 任务：Q21/Q22；分组：G0；更新：2026-09-08 UTC。

## 场景与调查结论

首个真实负载固定为 verl + Megatron + vLLM，在单机 8×B200 上对 Qwen3.6-35B-A3B 做短程 GRPO。它接近 verl 官方 Qwen3.5 35B MoE 八卡脚本的 TP2/EP8、rollout TP8 形态；Qwen3.6 与该路径使用同一个 `Qwen3_5MoeForConditionalGeneration` 架构名，当前 verl registry 已登记，vLLM 官方 recipe 也明确支持。verl 尚无“Qwen3.6 + Megatron + vLLM”的同名端到端示例，所以 Q21 只确认源码、依赖、注册、容量和输入可执行；真正的模型构造、显存与训练循环由 Q22 证明。

输入选 `openai/gsm8k`：MIT、无需 gated access，规则 reward 可直接形成真实 GRPO 更新，不恢复业务造数。完整查询日期、源码 SHA、官方入口和未选方案见 [research.json](research.json)。

## 设置与判据

唯一配置是 [q21-b14-verl-qwen36-megatron-vllm.json](../../../../configs/infra/q21-b14-verl-qwen36-megatron-vllm.json)。verl 固定 `7cb65014d3a6c84f59458367df768999e4f36c67`；Modal runtime image 固定 `im-ddKz6toouSd10HPSoofnFY`。环境实际为 torch 2.11.0+cu130、Transformers 5.9.0、vLLM 0.24.0、Megatron-Core 0.18.0、Megatron-Bridge 0.5.2、Transformer Engine 2.16.1、mbridge 0.15.1。

模型固定 Volume `syncopate-home:/models/Qwen3.6-35B-A3B`、revision `995ad96eacd98c81ed38be0c5b274b04031597b0`，26 个分片共 71,903,776,776 字节；复用 B01 完整身份记录，本次再次检查 index 引用文件均非空。MTP 关闭，BF16，GDN 使用 padded BSHD；初始场景沿用官方 35B Megatron 示例的 `vanilla_mbridge=True`，它已被上游标为兼容旧路径，是否能在本模型完成构造仍是 Q22 的明确风险。

固定输入在 `syncopate-home:/data/infra-probes/B14/gsm8k-740312ad-seed20260909/`：train 从 7,473 条按 seed 20260909 固定抽 128 条，validation 从 1,319 条按 seed+1 固定抽 32 条。训练拟跑 4 次 global update，batch 32、每题 4 个 rollout、response 上限 1,024；生成上界 524,288 token。8×B200 硬超时 45 分钟，按 $0.001736/GPU 秒计算 GPU 费用上界 $37.4976，正常目标 30 分钟/$25；Q21 实际未申请 GPU。

Q22 最低健康闸：真实生成、至少两个非零 optimizer update、grad norm/有效 mask 非零、无 NaN、权重版本传给 rollout、四个周期正常退出。观测 generation/old-logprob/ref/update_actor/update_weights/step、有效 token 吞吐、response 长度、MoE 负载、显存/利用率/功耗和进程/rank 身份；不以 GSM8K 正确率作为本轮门槛。

## 画像、改动与测试结果

| run-id / UTC | 做了什么 | 结果与关键读数 | 证据索引 |
|---|---|---|---|
| `q21-cpu-20260908T165215Z` | 迭代镜像与输入前置 | 暴露并修正无效 Docker tag、tar 字节校验、Modal 控制 Python、双重建目录、旧证据路径和 Transformers 5 长度误读；错误数据已精确删除 | Volume 同名目录；失败 attempts 保留 |
| `q21-final-20260908T171904Z` | 固定镜像、模型、数据和命令；Modal CPU 回读 | PASS；train token min/mean/P50/P90/max=`53/83.67/81/111/213`，validation=`57/86.69/82/118/147`；全部低于 512 | [run index](runs/q21-final-20260908T171904Z.json) |

train Parquet SHA256 `59cc78450367adcfbc8270843d2abacf995f3ebe5e30e18bf54bcac55acfe653`；validation SHA256 `522ec35548d4cc9d085f7b3e8244f430f0ad96bbdbc10277f42f6869f633b70e`。回下载后独立计算与 Volume manifest 一致。

## 结论与下一步

Q21 已把“跑什么”冻结成一份可执行配置：源码、依赖镜像、模型、输入、并行切法、真实命令、预算和观测均有唯一身份，CPU 前置通过。这里没有训练性能结论，也没有证明 Qwen3.6 的 Megatron 模型构造、vLLM 权重同步或 8 卡显存一定通过；这些正是 Q22 第一次真实 GPU 运行要回答的内容。

Q21 GPU 费用为 $0；CPU/镜像构建实际账单未读取，不用墙钟估算冒充结算。冻结 Parquet、HF 小缓存、最终与失败审计保留；两次错误 Parquet 已删除并复核后重建。未产生 checkpoint，未创建 upstream DRAFT。
