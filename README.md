# Pony Agent

Pony Agent 是从 Hermes Agent 演进而来的轻量 Agent Kernel。它保留 Hermes 在
provider、工具、审批、Profile、消息网关和 MCP 等外围积累的实现，重新建立一个
小型、可测试、async-first 的消息与工具循环。

当前版本是 `0.1.0a1`。CLI oneshot 已默认使用 Pony Kernel；交互式 CLI、Gateway、
Bedrock、ACP 和 Codex App Server 仍暂时运行 legacy loop。

- **仓库：** <https://github.com/ump45nose/pony-agent>
- **上游项目：** <https://github.com/NousResearch/hermes-agent>
- **许可证：** MIT，见 [LICENSE](LICENSE)

## 为什么是一个新 Kernel？

一个基础 agent loop 并不需要框架：接收消息、流式请求模型、执行工具，再把工具
结果交回模型即可。真正昂贵的是 provider 协议差异、reasoning/signature replay、
缓存与凭据刷新、工具审批、secret scope、Profile、平台重连和 MCP OAuth。

Pony 因此采用绞杀式迁移：

```text
pony -z
  -> AgentKernel
       -> ProviderAdapter  -> Chat / Responses / Anthropic / native Gemini
       -> ToolRuntime      -> existing registry / ACL / approval / receipts
       -> ContextPolicy    -> one bounded compaction attempt
       -> SessionStore     -> ~/.pony/kernel.db
       -> KernelEvent      -> streaming UI / future Gateway adapter
```

纯 kernel 位于 `pony_agent/core/`。它不导入 provider SDK、SQLite、Hermes CLI 或
具体工具；这些能力均通过窄接口注入。

## 从源码运行

需要 Python `>=3.11,<3.14` 和 [uv](https://docs.astral.sh/uv/)。Alpha 阶段尚未提供
独立安装器，请从源码运行：

```bash
git clone https://github.com/ump45nose/pony-agent.git
cd pony-agent
uv sync --extra all

uv run pony --help
uv run pony setup
uv run pony -z "Inspect this repository and summarize it"
```

显式回到 legacy loop：

```bash
uv run pony -z --agent-core legacy "your prompt"
```

Pony Kernel 遇到尚未迁移的 Bedrock、ACP 或 Codex App Server 时会明确失败并给出
上述回退方式，不会静默改变执行核心。

## 名称与数据硬切

Pony 不自动读取或迁移 Hermes 的公开入口：

| 项目 | Pony |
| --- | --- |
| Distribution | `pony-agent` |
| CLI | `pony`, `pony-agent`, `pony-acp` |
| 默认目录 | `~/.pony` |
| 环境变量前缀 | `PONY_*` |
| Profile | `~/.pony/profiles/<name>/` |
| Kernel 会话库 | `~/.pony/kernel.db` |
| Legacy 会话库 | `~/.pony/state.db` |
| 插件 entry point | `pony_agent.plugins` |

标准 provider 变量，例如 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`，不因项目改名而
变化。旧 `hermes` 命令、`HERMES_*` 和 `~/.hermes` 不提供别名或自动迁移。

## Kernel API

```python
from pony_agent.core import AgentKernel, KernelConfig

session = kernel.open_session(KernelConfig(model="provider/model"))
await session.submit("hello")

async for event in session.events():
    if event.kind == "text.delta":
        render(event.payload["delta"])
    if event.kind in {"run.completed", "run.failed"}:
        break
```

公开控制面包括 `submit`、`steer`、`follow_up`、`cancel` 和异步事件流。工具执行
结果分为 model content、UI details、receipt 和 effect，kernel 不解释具体工具
语义，也不会自动重放可能产生副作用的工具。

## 当前协议范围

| Provider wire protocol | Pony Kernel |
| --- | --- |
| OpenAI Chat Completions | 已接入 |
| OpenAI Responses / Codex Responses | 已接入 |
| Anthropic Messages | 已接入 |
| Native Gemini | 已接入 |
| Bedrock Converse | legacy only |
| ACP / Codex App Server | legacy only |

“已接入”表示 adapter 和事件契约已实现；某个 provider 是否端到端可用仍取决于
本机是否配置了有效凭据，并应通过一次真实文本流和一次真实 tool call 验证。

## 安全边界

- Pony Kernel 不取代工具 ACL、审批、secret scope 或 Profile 隔离。
- Provider opaque state 可以写入 `kernel.db`，API key、token、password、cookie 和
  Authorization 值会在事件入库前按键脱敏。
- 工具 side effect、receipt 和结果状态由现有 ToolRuntime 维护。
- `kernel.db` 使用 WAL；事件追加和查询投影在同一事务内提交。
- Gateway、Kanban、Episode 与平台 delivery 仍位于 kernel 之外。

## 开发验证

本项目只要求与改动直接相关的编译、导入和单路径业务冒烟；不要默认运行大范围
回归套件。Kernel 改动至少验证：

1. provider 文本流到 `text.delta`；
2. 一次 tool call 到 receipt/effect 持久化；
3. cancel 产生 partial checkpoint；
4. 关闭后可以从 `kernel.db` 重建消息；
5. 未支持协议给出 `--agent-core legacy` 提示。

## 上游归属

Pony 保留 Hermes Agent 的完整 Git ancestry、发布 tags、MIT License 和原作者
版权。Legacy 模块在迁移期间仍保留原有 Hermes 内部命名；这不表示存在公开的
Hermes 命令或数据兼容层。
