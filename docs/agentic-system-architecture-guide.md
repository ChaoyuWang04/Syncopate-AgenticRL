# Agentic System Architecture Guide · 从模型能力到可靠行动

> 本文是 Syncopate、Agent Framework Lab、Harness Lab 与 Sandbox RL/MOPD Lab 的跨项目架构指南。它不记录任何一条线的当前进度，不替代各 Lab 的 TASKS、计划、Gate 或证据。
>
> 它只回答长期设计问题：**一个新能力应该放在哪一层；何时使用确定性工作流、模型路由、专用模型、训练或 MOPD；怎样让训练、编排、可靠执行、沙盒和控制面形成闭环，而不是一堆彼此独立的组件。**

---

## 1. 核心判断：系统的“大脑”不是单一组件

完整 Agent 系统没有一个包揽全部职责的中央大脑。它由三类决策共同组成：

1. **模型智能**：理解模糊目标、推理、生成候选、处理非结构化信息；
2. **编排智能**：决定先做什么、调用谁、带什么上下文、什么时候停止和验收；
3. **系统控制**：决定谁有权限、资源够不够、动作能否执行、失败怎样恢复、证据如何保存。

如果所有决定都交给模型，系统不可控；如果全部写死在代码里，系统失去处理开放问题的能力。合理设计是在两者之间移动边界，并用证据决定边界应该放在哪里。

## 2. 六层架构与唯一职责

```text
┌───────────────────────────────────────────────────────────────┐
│ 6. Evaluation / Data / Learning                              │
│ 评测集、回放、失败归因、SFT、RL、OPD、MOPD                    │
├───────────────────────────────────────────────────────────────┤
│ 5. Control Plane                                              │
│ 身份、租户、连接、授权、审批、配额、Registry、策略             │
├───────────────────────────────────────────────────────────────┤
│ 4. Serving Harness / Run Plane                                │
│ run、outbox、queue、lease、retry、fencing、event、SLO          │
├───────────────────────────────────────────────────────────────┤
│ 3. Agent Framework / Model & Tool Gateways                    │
│ 目标、计划、图状态、路由、上下文、工具提案、验收               │
├───────────────────────────────────────────────────────────────┤
│ 2. Runtime / Sandbox / Computer Use                           │
│ 工具执行、文件/进程/浏览器环境、资源隔离、世界状态             │
├───────────────────────────────────────────────────────────────┤
│ 1. Model & Compute Infra                                      │
│ base/SFT/RL 模型、训练/推理引擎、GPU、通信、kernel             │
└───────────────────────────────────────────────────────────────┘
```

| 层 | 它决定什么 | 它不应决定什么 |
|---|---|---|
| Model/Infra | “对这个输入给出什么预测或动作候选” | 用户是否有权限、外部动作是否已成功 |
| Runtime/Sandbox | “工具在哪里执行，返回了什么真实 observation” | 为什么要做这个业务动作 |
| Agent Framework | “目标如何拆解，下一步调用模型还是工具，怎样验收” | 凭据明文、全局资源配额、可靠排队 |
| Harness | “任务怎样不丢、重试、恢复、发事件” | 任务的语义计划和模型能力判断 |
| Control Plane | “谁能用什么版本和多少资源” | 逐 token 推理、任意 shell、业务答案 |
| Evaluation/Learning | “哪些轨迹可信，哪里需要改进，是否值得训练” | 在线直接绕过审批改权重或策略 |

## 3. 两条主链和一个反馈环

### 3.1 在线执行链

```text
用户请求
  → Identity / RunPolicy
  → Harness 接受并持久化 run
  → Agent Framework 理解目标并制定验收计划
  → Model Gateway 选择 Generalist / Specialist / Teacher
  → Framework 形成 ToolCall proposal
  → Control Plane 校验 Grant / Approval / Quota
  → Tool Gateway 获取短期凭据
  → Sandbox / Computer Use 执行
  → 验证真实世界状态
  → Harness 保存终态、事件和 Capability Receipt
  → 用户得到自然语言答案
```

### 3.2 离线学习链

```text
真实 run 与 observation
  → M4 重建完整轨迹
  → 区分模型 / 编排 / 工具 / Harness / 环境故障
  → 脱敏、去污染、质量与资格 Gate
  → 冻结评测集与回放世界
  → 比较 prompt / route / model / policy / checkpoint
  → 只有证据支持时才进入 SFT / RL / OPD / MOPD
  → 新候选先 shadow / replay
  → 通过后再小流量上线
```

### 3.3 关键反馈环

Model Gateway 不只是请求代理，它同时是在线实验和数据生产平面：

```text
Gateway 路由多个教师
  → 同一任务产生带 teacher/model/policy 身份的候选
  → Tool/Sandbox 提供真实 observation
  → Verifier 判断任务是否真实完成
  → M4 排除环境与 Harness 故障
  → 得到 teacher-tagged、world-grounded trajectories
  → 再决定是否 MOPD 到单一学生
```

因此正确顺序通常是：**先路由、先观察、先积累真实差异，再蒸馏。** MOPD 不是发现教师分工的方法，而是压缩已经被证明有价值的分工。

---

## 4. 能力应该放在哪一层：七级决策阶梯

每遇到一个“模型做不好”的问题，按顺序问。只有上一层不能合理解决，才进入下一层。

### 第 1 级：确定性规则或验证器

适合：权限、金额范围、Schema、终态断言、幂等、审批、引用完整性、循环上限。

判断：如果规则能被准确写下并低成本执行，就不要让模型猜。

### 第 2 级：工具、检索或世界状态

适合：当前余额、日历、文件、网页、数据库、内部政策、最新事实。

判断：模型缺的是事实，不是能力。让它读取事实，而不是把会变化的事实训练进权重。

### 第 3 级：上下文工程

适合：模型不知道当前用户、可用工具、历史结论、输出格式或局部业务说明。

判断：先检查模型实际看到的 prompt、messages、tools、observation 和截断，再归因模型。

上下文分为：

- transient model context：本次调用看到的指令、消息、工具和格式；
- graph state：当前 thread 的短期状态；
- store/memory：跨 thread 的长期信息；
- runtime context：身份、连接引用、权限摘要等不应写入聊天历史的配置。

LangChain 也把“给模型正确的信息与工具”视为 Agent 可靠性的中心问题。参考：
<https://docs.langchain.com/oss/python/langchain/context-engineering>

### 第 4 级：Framework 编排或路由

适合：某些请求需要额外步骤、专用上下文、专用工具、并行分析或人工审批。

判断：能力存在于某个模型/工具，只是并非每次都该调用。用 graph、router、subagent 或 evaluator 组织。

### 第 5 级：专用模型或 Adapter

适合：特定垂类有稳定、重复、可测的能力需求，通用模型通过 prompt/工具仍明显不足。

判断：先保持独立 endpoint/adapter，便于升级、回滚和度量边际价值。

### 第 6 级：SFT、RL 或 OPD

适合：需要稳定改变模型行为、工具使用策略、长程决策或领域输出分布。

判断：必须有可重放评测、可靠环境、训练输入与线上同形、清晰 reward/verifier。否则训练会吸收系统错误。

### 第 7 级：MOPD 合版

适合：多个教师的能力已被证明互补，但在线路由成本、延迟、隐私、离线部署或运维复杂度不可接受。

判断：只有学生在所有关键 slice、域外集和安全 Gate 上达到预注册保真度，才晋级合版。

---

## 5. Workflow、Router、Subagent、Handoff 怎样选择

| 模式 | 适合 | 不适合 |
|---|---|---|
| Deterministic workflow | 路径和验收已知、风险高、需复现 | 开放式未知步骤全部写死 |
| Router | 一次分类后进入不同垂类；可并行 | 需要长程、多轮重新决策 |
| Supervisor + subagent/tools | 主 Agent 统一负责，专家只提供局部结果 | 专家必须长期直接与用户互动 |
| Handoff | 对话阶段明确切换，专家接管多轮 | 只需一次局部判断却传递全历史 |
| Orchestrator-worker | 子任务数量运行时才知道，可并行汇总 | 简单线性任务 |
| Evaluator-optimizer | 成功标准明确、允许有限迭代 | 无确定标准或无限自我反思 |

原则：先单 Agent + 小工具面；当出现明确的上下文污染、能力分区、并行收益或独立维护需求时，再增加 Agent。

LangGraph 官方提供 routing、parallelization、orchestrator-worker 和 evaluator-optimizer 等模式，但它们是不同问题的工具，不应为了“像 Agent 系统”而全部叠加。参考：
<https://docs.langchain.com/oss/python/langgraph/workflows-agents>

---

## 6. 什么时候路由，什么时候训练，什么时候合版

### 6.1 决策矩阵

| 现象 | 首选动作 | 原因 |
|---|---|---|
| 信息会频繁变化 | Tool/RAG | 权重很快过时 |
| 规则明确且高风险 | 确定性 policy/verifier | 可证明、可审计 |
| 少数请求需要垂类能力 | Model Gateway 路由 | 避免全部流量付专家成本 |
| 专家频繁独立升级 | 保持路由 | 可单独发布和回滚 |
| 同模型只需不同少量知识 | skill/prompt/context | 成本最低 |
| 通用模型行为长期不稳 | SFT/adapter | 改变稳定输出分布 |
| 需要从真实反馈学习策略 | RL/OPD | 目标来自交互和 verifier |
| 多专家高频调用导致成本/延迟问题 | 评估 MOPD | 压缩在线组合 |
| 必须单模型、离线或私有部署 | adapter merge/MOPD | 运行约束驱动合版 |
| 教师在关键样本上冲突 | 继续路由并挖掘分歧 | 平均会抹去重要边界 |
| 学生域外退化或安全回归 | 不合版 | 不能用平均分掩盖灾难遗忘 |

### 6.2 路由的成本函数

不要只问“哪个模型分数高”，而应比较：

```text
route_value
  = task_success_gain
  - latency_cost
  - token_and_money_cost
  - provider_failure_cost
  - orchestration_complexity
  - privacy_and_compliance_cost
```

不同业务切片的权重不同。高风险写操作可能愿意支付强验证模型成本；普通问候不值得调用 Specialist。

### 6.3 MOPD 的晋级条件

至少同时具备：

1. 教师的能力边界在真实/回放任务上可复现；
2. Gateway 有 teacher、route、policy、world 和 verifier 身份；
3. 非模型故障已从训练资格中剔除；
4. 学生的 on-policy 状态分布被覆盖，而非只模仿教师离线答案；
5. 关键域、混合域、域外和安全集均有 Gate；
6. 合版能解决一个实测问题：成本、延迟、离线、隐私或维护；
7. 保留原路由架构作为回滚和对照。

如果没有第 6 条，MOPD 可能是有趣研究，但不是架构必需品。

---

## 7. 可复用的组合设计模式

### 7.1 Specialist as Proposer

```text
Specialist → ActionProposal
Policy Engine → allow / deny / approval_required
Human → approve / reject
Tool Gateway → execute
Verifier → confirm world state
```

专用模型提供业务判断，但不拥有权限和凭据。这让小型垂类模型即使对话能力较差，也能作为可靠的结构化能力模块使用。

### 7.2 Shadow Teacher

线上主路径照常执行；候选教师读取相同的脱敏 state，在不调用真实写工具的影子路径给出候选。记录：

- 是否会选择相同工具；
- 参数差异；
- verifier 预测；
- token、延迟、成本；
- 与真实结果的反事实差异。

Shadow 不能共享真实副作用幂等键，也不能误写为“任务成功”。

### 7.3 Counterfactual Replay

把 M4 保存的 case、pre-state、tool observations、fault schedule 和 assertions 冻结，让不同模型或策略在相同世界上重走。工具写操作由 simulator 接管，不重复触碰真实 API。

用途：

- 比较 Generalist-only 与 routed；
- 比较旧/新 prompt；
- 测试不同 Gateway policy；
- 发现模型改进其实来自环境变化的假象。

### 7.4 Disagreement Mining

多个教师一致的高置信、已验证轨迹适合成为蒸馏候选；教师分歧不应简单平均，应进入：

1. 人工审查；
2. 新评测切片；
3. 路由边界修订；
4. verifier 加固；
5. 必要时保留专家分流。

教师分歧是最有信息量的数据之一。

### 7.5 Capability Receipt

每个终态都生成收据：

- principal/tenant 的安全摘要；
- run policy hash；
- graph/agent/model/prompt/tool/sandbox 版本；
- route reason 和 fallback；
- approval 与 idempotency receipt；
- verifier/assertion 结果；
- trace 与 trajectory 引用。

Receipt 让“模型说完成了”变成“系统能够证明发生了什么”。

### 7.6 Failure-aware Learning

轨迹失败必须先判 owner：

```text
model | orchestration | policy | harness | provider | tool | sandbox | verifier | fixture
```

只有与模型行为相关、世界证据完整的样本才进入行为训练候选。将 timeout、容器失败或错误 verifier 当成 reward 0，会训练模型回避本来正确但环境容易坏的长任务。

### 7.7 Escalation Ladder

```text
deterministic code
  → small generalist
  → vertical specialist
  → strong generalist/verifier
  → human approval/review
```

升级条件可以来自风险、uncertainty、validator failure、教师分歧、预算和 deadline。降级也要显式：强模型不可用时，不得静默让弱模型执行高风险动作。

### 7.8 Branch-and-Verify

对高价值但可验证的任务，让便宜模型产生多个候选，强 verifier 或确定性测试选择一个。只允许胜出候选进入审批和真实执行。

这比让强模型包办每一步更省成本，也比多候选全部执行更安全。

### 7.9 Distill the Expensive Residual

不要蒸馏全部流量。Gateway 先找出：

- 高频升级到昂贵教师的切片；
- Specialist 带来稳定增量的切片；
- latency/cost 占比最高的切片；
- 能被可靠 verifier 判断的切片。

只蒸馏这些“昂贵残差”，常比做一个万能学生更容易证明价值。

### 7.10 Adapter Before Merge

先把领域能力保存为独立 adapter/endpoint，并通过 Gateway 做显式分流。只有多个 adapter 的能力边界和冲突被评测清楚，才考虑 merge 或 MOPD。

独立阶段提供了回滚、A/B、数据归因和教师身份；过早 merge 会把这些证据全部抹平。

### 7.11 Two Clocks

- **在线时钟（毫秒到分钟）**：Framework 根据本次目标、风险和状态选择模型/工具；
- **离线时钟（天到周）**：评测系统聚合真实证据，决定是否更新 prompt、router、adapter 或权重。

线上 Agent 不应因为一次失败自行改长期策略或权重。任何学习都先进入隔离评测和发布流程。

### 7.12 Gateway as Experiment Plane

Model Gateway 应同时支持：

- policy-aware routing；
- stable model identity；
- canary/shadow；
- fallback 与 circuit breaker；
- token/费用/延迟计量；
- teacher disagreement 记录；
- cohort 分桶；
- trajectory lineage。

但 Gateway 不判断业务任务是否完成；最终成功必须由 Tool/Sandbox 的真实结果和 verifier 确认。

### 7.13 Contract Parity, Environment Diversity

训练、离线 replay、Framework fixture 和生产 Sandbox 可以使用不同物理环境，但必须共享版本化契约。反过来，即使它们运行在同一台机器上，工具名、错误语义或 observation 格式不同，也不叫训推一致。

### 7.14 Evidence-gated Autonomy

Agent 的自治等级不由“模型看起来聪明”决定，而由证据逐层解锁：

```text
answer only
  → read tools
  → write proposal
  → human-approved write
  → bounded auto-write
  → broader autonomy
```

每次升级必须有独立评测、可回滚 policy 和更严格 receipt；低风险读能力通过不自动解锁高风险写能力。

---

## 8. 六个跨层核心契约

最终融合主线前，应先稳定以下契约，而不是先搬代码：

### 8.1 `RunPolicy`

身份、Agent、模型候选、工具权限、Connection、预算、Sandbox profile、privacy 和 revocation 的不可变快照。

### 8.2 `ModelRequest / ModelResponse`

消息、工具、结构化输出、预算、provider/model identity、usage、latency、finish/error。不得把 provider 私有格式泄漏进 Graph state。

### 8.3 `AgentState / GraphRunResult`

目标、计划、route、proposal、approval、receipts、verification、terminal status 和 lineage。

### 8.4 `ToolCall / ToolResult`

规范化参数、风险级、connection reference、idempotency key、observation、error taxonomy 和 side-effect receipt。

### 8.5 `SandboxProfile / SandboxLease`

provider、镜像/环境、CPU/内存/GPU、网络、挂载、TTL、fencing、create/execute/terminate/reconcile 语义。

### 8.6 `TrajectoryEnvelope`

case/world、policy、model turns、tool receipts、faults、failure owner、verifier、privacy、training eligibility 和全部版本 hash。

这些契约必须有正负 fixture、版本策略和 consumer-driven contract tests。只有 Schema 文件存在而没有真实消费者不算集成。

---

## 9. 可观测性应该观察“决策”，不只观察服务

### 9.1 四种不同信号

| 信号 | 回答的问题 |
|---|---|
| Metrics | 系统是否整体变慢、积压、失败、超预算 |
| Logs | 某组件具体发生了什么 |
| Traces | 一次 run 如何跨模型、图节点、工具和沙盒传播 |
| Semantic receipts/trajectories | 为什么选这个模型/动作，世界是否真的改变 |

### 9.2 必须能回答的十个问题

对任意 run，系统应能回答：

1. 用户最终目标是什么？
2. 哪些成功条件被预先写下？
3. 为什么选择这个图和模型？
4. Specialist 看到了哪些上下文？
5. 模型提出了什么动作？
6. 哪条策略允许或拒绝了它？
7. 谁批准了什么精确 proposal？
8. Tool/Sandbox 实际执行了什么，是否恰好一次？
9. 验收读取了什么真实世界状态？
10. 如果失败，责任属于模型、编排还是环境？

缺少其中任一项，数据回流时就容易把错误学习进模型。

---

## 10. 从独立 Lab 到主线的融合顺序

本项目施工时序仍以各权威计划为准：先完成 Harness M4，再完成 Learn L0–L8，最后才启动产品 Control Plane。下面描述的是这些 Lab 已经分别完成以后，**把已存在的能力逐项接回主线**的集成顺序，不是让 Control Plane 排在 M4 建设之前。

融合应按契约和风险递增，而不是把目录一次合并：

1. **Schema/fixture**：主线只读消费版本化契约；
2. **Model Gateway adapter**：先接模型调用和 receipt，不改变工具执行；
3. **Framework graph**：先 read-only 场景，与旧 AgentLoop A/B；
4. **Harness adapter**：一个 run 对应一个 graph thread，验证双层恢复；
5. **Sandbox adapter**：保持工具契约，先 read 后 write；
6. **Control Plane policy**：身份、Connection、Grant、Approval 和 Quota；
7. **重接 M4 trajectory contract**：让已完成的 M4 轨迹新增 policy/approval/receipt lineage，并重新验证失败归因和回放；
8. **shadow/canary**：候选模型或策略不产生高风险副作用；
9. **有限生产流量**：逐 cohort 晋级，可一键回滚；
10. **训练/MOPD**：最后才让已验证轨迹影响权重。

每一步都保留旧路径作为对照，直到新路径在同一尺子下通过。融合不是“Lab 完成后的代码搬家”，而是逐个替换可证明的契约实现。

## 11. 应避免的反模式

- 一个模型同时负责规划、授权、执行和自我验收；
- 为了显示“多 Agent”而给每个步骤创造人格；
- Specialist 获取全部会话、全部工具和全部凭据；
- Framework、Harness、Sandbox 各有一套互不映射的 run ID；
- LangGraph checkpoint 被误称为外部副作用 exactly-once；
- Control Plane 每一步都在线查询，失联时整个 run 无法收敛；
- Gateway 按模糊模型别名路由，无法还原真实 revision；
- 只记录最终答案，不记录 route、tool observation 和 verifier；
- 把 Sentry issue、HTTP 500 或超时直接当模型负样本；
- 教师未证明互补，就先做 MOPD；
- 只看平均分，掩盖安全、域外或长任务切片退化；
- 训练环境和部署环境使用同一镜像，就宣称训推一致；
- 用更大的模型掩盖错误上下文、错误工具或错误 verifier；
- 一次性把三个 Lab 合回主线，没有 staged migration 和 rollback。

## 12. 新能力设计记录模板

以后提出新能力时，先回答以下问题：

```markdown
## Capability: <名称>

- 用户价值：
- 可观测成功条件：
- 当前失败证据：
- 缺的是事实、上下文、编排还是模型能力：
- 最低可行放置层：规则 / 工具 / 上下文 / Framework / 专用模型 / 训练 / MOPD
- 为什么更低层不能解决：
- 所需契约：
- 安全与审批级别：
- 评测切片和负向 case：
- 失败 owner 分类：
- shadow/replay 方案：
- 费用、延迟和资源上限：
- 回滚路径：
- 何时升级到下一层：
- 何时删除或简化：
```

这个模板的目的不是增加流程，而是防止把所有问题都误判成“再训练一个模型”或“再增加一个 Agent”。

## 13. 当前项目的推荐长期结构

```text
Syncopate mainline
  ├── Training / Inference Infra
  │     └── 训练引擎、推理引擎、性能、权重交接
  ├── Model Development
  │     └── base、SFT、RL、OPD、MOPD、模型评测
  ├── Agent Framework
  │     └── LangGraph、状态、计划、路由、上下文、验收
  ├── Harness / Run Plane
  │     └── API、DB、queue、lease、recovery、SSE、telemetry
  ├── Control Plane
  │     └── identity、connection、grant、registry、quota、policy
  ├── Runtime / Sandbox
  │     └── tools、browser、computer use、filesystem、isolation
  └── Evaluation / Data Flywheel
        └── trajectory、replay、failure attribution、learning eligibility
```

独立 Lab 阶段允许各模块快速试错；回归主线时，主线只接纳已经通过契约、回归、资源和安全 Gate 的部分。未验证的 Lab 机制不自动成为生产默认。
