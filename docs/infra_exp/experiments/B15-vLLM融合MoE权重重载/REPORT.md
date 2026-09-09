# infra-probes/B15 · vLLM 融合 MoE 权重重载

状态：**Q23 背景与根因已冻结；修法 A/B 未开始。** 分组：G4/G6；更新：2026-09-09 UTC。

## 为什么做

B14 的真实 8×B200 入口已经通过 Megatron TP2/EP8 模型构造和 vLLM DP2/TP4/EP8 MoE kernel autotune，却在第一次把训练权重交给 rollout 时失败。这是训练框架和推理引擎之间的真实接缝，直接阻塞同一场景的生成、更新与性能画像。

固定环境的 vLLM 0.24.0 中，Qwen3.5 专用 loader 把 `[experts, out, in]` 融合输入逐专家切成二维，再装入 modular RoutedExperts 的三维 `w13_weight` 目标，并以 `shard_dim=0` 调用通用 loader。通用 loader 明确把三维目标的第0维视为专家轴，只允许数据维1/2，因此硬失败。当前 vLLM 主线已经删除这条 Qwen 专用逐专家逻辑，改由 RoutedExperts 统一识别和拆分融合权重；说明修复方向存在，但尚未证明与本项目的 verl、Torch、Megatron Bridge 和 B200 锁兼容。

## Q23 计划与判据

1. 先冻结能同时满足 verl 当前接口和 B200/CUDA 13 的最新稳定 vLLM 版本；若直接升级牵动 Torch 或 verl 锁，登记完整依赖差异，不在生产 venv 临时覆盖。
2. 从 B14 保存一层真实 Megatron 融合 expert 权重，建立 vLLM 0.24.0 负对照与候选版本/最小实验性修法正对照。逐字检查权重名、shape、专家映射、每个 EP rank 的本地 expert 和 reload 前后参数摘要。
3. 单层通过后回 B14 原入口，只要求第一次 `wake_up/update_weights` 成功；随后再恢复 Q22 的四步训练与 profiler 画像。

完成判据：EP8 所有 rank 都装入正确本地专家；初始 HF load 与同权重 reload 的参数身份一致；真实权重交接至少成功一次。若候选版本需要破坏 verl 的冻结依赖或回归失败，停止并登记为 upstream DRAFT，不用降低并行规模掩盖问题。

原始触发证据：`/vol/_audit/infra-probes/B14/q22-smoke-09-20260909T1251Z/attempt-82ac55d5d4134b9dab63cd3bdc7b0fe4/`。详细检索见 [research.json](research.json)。
