# infra-probes/B15 · vLLM 融合 MoE 权重重载

状态：**Q23 已改为最新版全栈重建；旧版修法A/B取消。** 分组：G0/G4/G6；更新：2026-09-09 UTC。

## 为什么做

B14 的真实 8×B200 入口已经通过 Megatron TP2/EP8 模型构造和 vLLM DP2/TP4/EP8 MoE kernel autotune，却在第一次把训练权重交给 rollout 时失败。复查后确认那次运行直接继承了 verl 镜像中的 vLLM 0.24.0，而不是每个组件各自的最新版；因此它能说明旧组合在哪里失败，不能说明当前栈仍有同一问题。

固定环境的 vLLM 0.24.0 中，Qwen3.5 专用 loader 把 `[experts, out, in]` 融合输入逐专家切成二维，再装入 modular RoutedExperts 的三维 `w13_weight` 目标，并以 `shard_dim=0` 调用通用 loader。通用 loader 明确把三维目标的第0维视为专家轴，只允许数据维1/2，因此硬失败。当前 vLLM 主线已经删除这条 Qwen 专用逐专家逻辑，改由 RoutedExperts 统一识别和拆分融合权重。Q23要先判断这个旧问题在最新版全栈里是否已经消失，而不是给0.24.0补丁。

## Q23 计划与判据

1. 分别从每个项目的官方发布源解析 verl、PyTorch、Transformers、vLLM、Megatron-Core、Megatron-Bridge、Transformer Engine、mbridge、FLA以及所选CUDA基础镜像的最新稳定版，同时记录各仓库最新 `main` SHA。不能让任一框架的镜像或lock替其他组件选版本。
2. 在隔离的新镜像安装这组最新版，先做依赖解析、import、API契约和B200反向健康检查。若最新版互不兼容，保存精确求解冲突/接口错误并停止；不通过降级构造“兼容版本”。版本清单解析超过6小时后，GPU调度前重新解析。
3. 最新版全栈健康后，复用B14的模型、输入、TP2/EP8与DP2/TP4/EP8布局重新进入真实权重交接。vLLM 0.24.0只作为历史负对照；最新版若仍出现同类错误，再保存一层真实融合expert权重做逐rank身份定位。
4. `wake_up/update_weights`通过后恢复Q22四步训练与profiler画像；旧错误不复现则直接关闭，不为已经修复的旧版本写本地补丁。

完成判据：所有登记组件实装版等于各自最新稳定版、都有最新 `main` SHA，且清单未超过6小时；最新版组合通过CPU与B200健康闸；真实权重交接至少成功一次。若最新版组合不兼容，Q23以明确兼容性阻塞结项或派生有界任务，不能降低版本或并行规模掩盖问题。

原始触发证据：`/vol/_audit/infra-probes/B14/q22-smoke-09-20260909T1251Z/attempt-82ac55d5d4134b9dab63cd3bdc7b0fe4/`。详细检索见 [research.json](research.json)。
