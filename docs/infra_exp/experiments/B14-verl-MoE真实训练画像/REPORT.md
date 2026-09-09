# infra-probes/B14 · verl MoE 真实训练画像

状态：**历史运行已保留；依赖栈已判过期，等待 Q23 重建最新版全栈。** 任务：Q21/Q22；分组：G0；更新：2026-09-09 UTC。

## 场景与调查结论

首个真实负载固定为 verl + Megatron + vLLM，在单机 8×B200 上对 Qwen3.6-35B-A3B 做短程 GRPO。输入选 `openai/gsm8k`：MIT、无需 gated access，规则 reward 可直接形成真实 GRPO 更新，不恢复业务造数。完整查询日期、源码 SHA、官方入口和未选方案见 [research.json](research.json)。

Q21 当时把源码、镜像、模型、输入、真实命令、预算和观测冻结成唯一配置 [q21-b14-verl-qwen36-megatron-vllm.json](../../../../configs/infra/q21-b14-verl-qwen36-megatron-vllm.json)。它错误地以 verl 镜像为版本锚点：verl 为 `7cb65014d3a6c84f59458367df768999e4f36c67`，实际环境为 torch 2.11.0+cu130、Transformers 5.9.0、vLLM 0.24.0、Megatron-Core 0.18.0、Megatron-Bridge 0.5.2、Transformer Engine 2.16.1、mbridge 0.15.1。该组合现在只作历史负对照，不能作为当前baseline或upstream候选。模型 revision `995ad96eacd98c81ed38be0c5b274b04031597b0`，26 个分片共 71,903,776,776 字节。

固定输入在 `syncopate-home:/data/infra-probes/B14/gsm8k-740312ad-seed20260909/`：train 128 条，validation 32 条。训练计划为 4 次 global update，batch 32、每题 4 个 rollout、prompt/response 上限 512/1,024。8×B200 硬超时 45 分钟；Q22 最低健康闸是：真实生成、至少两个非零 optimizer update、grad norm/有效 mask 非零、无 NaN、权重版本传给 rollout、四个周期正常退出。

## 可观测性与显存设置

整场只用 1 秒一次的 `nvidia-smi`；Torch profiler 预注册为 global step 2、actor rank 0、CPU/CUDA/shape，steps 3/4 才用于无 profiler 的基线。nsys 2025.6.1 已确认存在，但本轮不包住训练；DCGM CLI 不存在。vLLM 显存目标固定为 35%（约 63 GB/GPU），`max_model_len=1536`，CUDA graph 只捕获 1/2/4/8/16/32，避免模型原生 262,144 context 和大图池吞掉无关显存。

训练侧采用 Megatron TP2/EP8。实际运行证明 rollout TP8 或 TP4×DP2 且 EP 关闭时，vLLM 会把 MoE expert 张量按 `TP×DP=8` 切分，512 宽的专家在单 rank 只剩 64，不能满足所选 FlashInfer TRTLLM BF16 MoE kernel 的 128 对齐。因此最终诊断布局固定为 attention DP2×TP4、expert EP8：8 张卡都参与，专家本身不再沿 TP 切成 64。

## 逐次结果

| run-id / UTC | 做了什么 | 结果与关键读数 |
|---|---|---|
| `q21-final-20260908T171904Z` | Modal CPU 回读配置、模型和输入 | PASS；train token min/mean/P50/P90/max=`53/83.67/81/111/213`，validation=`57/86.69/82/118/147`；全部低于 512。见 [run index](runs/q21-final-20260908T171904Z.json) |
| `q22-smoke` ～ `05` | 真实入口逐层启动 | 依次暴露 Hydra 新键语法、一次外部 Modal reset、旧 mbridge 参数、缺失 FLA 包；均在训练前结束，修正后不再复现 |
| `q22-smoke-06` | 首次完成 Megatron actor/ref 与 vLLM 初始化 | 发现 vLLM 继承模型原生 `max_model_len=262144`，显存计划不成立；主动收掉。峰值 66,356 MiB/GPU |
| `q22-smoke-07/08` | 固定 1536 context；比较 rollout TP8 与 TP4×DP2 | 两次都在 FlashInfer TRTLLM BF16 MoE autotune 失败：默认 ETP 令本地专家宽度为 64。DP2 本身没有改变这个事实。峰值 66,210 / 67,860 MiB/GPU |
| `q22-smoke-09` | vLLM attention DP2×TP4，expert EP8 | EP8 成功通过 kernel autotune；随后在 Megatron→vLLM 首次权重同步失败。Qwen3.5 专用 loader 把逐专家二维输入装入三维 `w13_weight`，以 `shard_dim=0` 调用通用 loader，报“3D tensor 只接受维 1/2”。峰值 120,220 MiB/GPU；14,664 个遥测样本无坏行；没有进入 global step，因此没有 Torch profile |

九次尝试的精确路径和终态见 [Q22 run index](runs/q22-20260909-investigation.json)。第9次原始证据在 `/vol/_audit/infra-probes/B14/q22-smoke-09-20260909T1251Z/attempt-82ac55d5d4134b9dab63cd3bdc7b0fe4/`。

## 结论与下一步

这次 smoke 已经提供了真实价值，但还不是速度 baseline。它证明 Qwen3.6 actor/reference 能在 Megatron TP2/EP8 上构造，vLLM 的 8 卡 EP kernel 能初始化，35% rollout 预留下峰值约 117.4 GiB/GPU，离 179.1 GiB 可见容量仍有约 61.7 GiB。长期平均 GPU 利用率低于 1.3%只表示绝大多数时间在初始化、编译、屏障和失败后等待，不能当训练 MFU。

历史组合里的直接失败点是框架接缝：Megatron Bridge 发送融合专家权重，vLLM 0.24.0 的 Qwen3.5 专用 reload 路径与 modular RoutedExperts 的目标布局不一致。当前 vLLM 主线已改为由 RoutedExperts 统一处理融合权重，所以不能把旧失败继续当成当前栈问题。Q23/B15 先分别更新所有直接组件到各自最新稳定版；最新组合若能完成交接，就关闭这个旧版本问题并恢复Q22，若仍失败才继续单层定位。

本轮没有完成生成、optimizer update、权重版本交接或 profiler step，不能声称训练健康或性能通过。当前配置已加 `blocked_stale_stack`，入口会在申请GPU前拒绝它。没有 checkpoint；失败日志和遥测保留在 Volume。第9次报错后主训练进程被 Ray 卡住，原 wrapper 不能自动退出；本地已增加终态 worker error 检测，后续会主动结束本实验进程组，避免无效占卡。实际 Modal 账单未读取，不用墙钟估算成费用。
