# Pony Agent

Pony Agent 是一个面向可组合 Agent 的 Python runtime。它把一次 agent 运行压缩为一条
async-first loop：接收消息、调用模型、按需执行工具、把工具结果继续交给模型，并以
事件流和持久化会话暴露给 CLI、Gateway 及其他宿主。

项目的重点不是把所有能力塞进核心，而是用窄接口连接 provider、tool runtime、context
policy、session store 和事件消费者。这样核心保持小、可测试，外围能力可以独立演进。

> 当前版本：`0.1.0a1`（Alpha）。接口和协议仍在迭代；未接入的 provider 或宿主会显式
> 报错，不会静默切换到另一条执行路径。

- **仓库：** <https://github.com/ump45nose/pony-agent>
- **Python：** `>=3.11,<3.14`
- **许可证：** MIT，见 [LICENSE](LICENSE)

## 项目特点

### 精简 loop

Kernel 只负责消息、模型、工具和事件之间的编排：

```text
submit / follow_up / steer
          │
          ▼
     provider stream
       ├─ text.delta / reasoning.delta
       ├─ tool.requested
       │    └─ scoped lookup → execute → receipt / effect
       └─ run.completed / run.failed
```

- `KernelSession` 提供 `submit`、`follow_up`、`steer`、`cancel` 和异步 `events()`；
- 工具调用可以批量执行，结果同时保留 model content、UI details、receipt 和 effect；
- 上下文只在需要时进行一次有边界的压缩，避免把上下文策略、provider SDK 或具体工具
  语义写死在 loop 中；
- 事件可持久化并在进程重启后重建会话，副作用工具不会被 Kernel 自动重放。

Kernel 的公开 API 通过依赖注入保持窄而稳定：

```python
from pony_agent.core import AgentKernel, KernelConfig

# provider、tools、store、context 由宿主选择具体实现。
kernel = AgentKernel(provider=provider, tools=tools, store=store, context=context)
session = kernel.open_session(KernelConfig(model="provider/model"))

await session.submit("检查这个仓库并给出一句话总结")
async for event in session.events():
    if event.kind == "text.delta":
        render(event.payload["delta"])
    if event.kind in {"run.completed", "run.failed"}:
        break

await session.close()
```

### Kanban：把多 Agent 协作放在 loop 之外

Kanban 是持久化的任务板和共享黑板，不是第二条隐藏的 agent loop。它为协作提供：

- 任务创建、领取、heartbeat、评论、附件、完成和阻塞等明确生命周期；
- 一个 dispatcher 负责在同一 board 上认领 ready task，避免多个 gateway 重复派发；
- `root → 并行 specialist workers → verifier → synthesizer` 的 swarm 拓扑；
- 任务评论和事件作为结构化协作记录，worker 不需要共享完整对话上下文；
- board 级隔离，以及按 profile/toolset 显式授予的 worker、orchestrator 能力。

因此，单个 Agent 仍然只运行精简 loop；Kanban 负责把多个独立 profile 组织成可观察、
可恢复的工作流。CLI、dashboard 和 slash 入口都可以接入同一块任务板。

### Multi-profile 下的 Multi-agent

一个 Gateway 可以根据 `platform`、`guild_id`、`chat_id`、`thread_id` 将消息路由到不同
profile。每个 profile 保持独立的 persona、memory、session 和 tool scope；Kanban 又可
以把不同任务分配给不同 profile，让 specialist 并行工作，再由 verifier/synthesizer
汇总结果。

这种设计把“多 Agent”拆成两个可独立控制的边界：

1. **路由边界：** 入站消息只进入匹配到的 profile；
2. **任务边界：** worker 只看到被分配的 board/task，完成、阻塞和 heartbeat 都写回任务板。

Profile routing 依赖多 profile runtime 开关；关闭时保持单 profile 行为，避免隐式改变
现有会话的状态空间。详见 [profile routing 文档](docs/profile-routing.md) 与
[多 Gateway 看板说明](docs/kanban/multi-gateway.md)。

### Tool / Skill 渐进式加载

工具和技能采用渐进式披露，而不是把全部能力永久注入每次模型请求：

- 基础 toolset 先按当前会话、profile 和 allowlist 收敛；可延迟工具在需要时通过 scoped
  tool search 查找并展开；
- skill bundle 可以一次显式加载多个技能，技能内容会经过启用状态、路径和缓存校验；
- 渐进式披露是可配置的运行策略，未启用的工具/技能不会因为“仓库里存在”就自动出现；
- 工具 schema 与 skill payload 分开处理，既减少 prompt footprint，也保留 ACL、审批和
  secret scope 的边界。

这让“能力很多”和“每一轮都很重”可以同时成立：常用路径保持短，长尾能力按需出现。

## 架构边界

```text
pony_agent/core/
  AgentKernel / KernelSession / KernelEvent
        │  narrow ports
        ├─ ProviderAdapter   → Chat / Responses / Anthropic / Gemini 等 wire protocol
        ├─ ToolRuntime       → registry / ACL / approval / receipts
        ├─ ContextPolicy     → bounded compaction
        └─ SessionStore      → event append + session rebuild

agent/、tools/、gateway/、plugins/、skills/
  能力适配、入口和扩展；不把实现细节反向塞回 core。
```

Kernel 目前优先用于 oneshot 路径；其他宿主通过适配层接入。Provider adapter 是否能端到
端运行，仍取决于本机配置的凭据和实际 wire protocol，建议用一次真实文本流和一次真实
tool call 做验证。

## 从源码运行

需要 Python `>=3.11,<3.14` 和 [uv](https://docs.astral.sh/uv/)。Alpha 阶段请直接从源码运行：

```bash
git clone https://github.com/ump45nose/pony-agent.git
cd pony-agent
uv sync --extra all

uv run pony --help
uv run pony setup
uv run pony -z "Inspect this repository and summarize it"
```

oneshot 默认使用 Pony Kernel；需要选择模型或 provider 时，沿用项目的 provider 配置和
环境变量即可，例如 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`。不要把凭据写入仓库或提交到
配置文件。

## 开发与验证

项目遵循“小改动、小验证”的节奏。修改 Kernel、adapter 或工具时，至少覆盖与改动直接
相关的一条路径：

1. 编译/导入通过；
2. provider 文本流能产生 `text.delta` 和终止事件；
3. 一次 tool call 能产生 receipt/effect；
4. cancel 能产生 partial checkpoint；
5. 关闭后可以从 session store 重建消息；
6. 未支持协议能给出明确的错误和下一步提示。

不要默认运行大范围回归套件；需要扩大验证范围时，请在变更说明中写清楚原因。

## 文档入口

- [Profile routing](docs/profile-routing.md)：按平台、服务器、频道和线程隔离 profile；
- [多 Gateway 看板](docs/kanban/multi-gateway.md)：dispatcher、board ownership 与多入口协作；
- `docs/`：会话生命周期、provider、Gateway 和扩展设计；
- [LICENSE](LICENSE)：MIT 许可证。

欢迎提交 issue、文档修订和小而可验证的改动。请优先保持 core 的边界，让新能力落在
adapter、tool、skill、plugin 或 Gateway 等合适的外围层。
