# code-review-agent · 代码审查助手

> ✅ 交付版（《软件工程与项目实践》第 1 次作业）。架构、接口规格与设计决策详见 [docs/Design.md](docs/Design.md)。

基于 LLM 的命令行代码审查工具。给它一段/一个/一组代码文件（任意文本型源码与配置文件，Python 为主），它以代码审查员身份输出分级问题清单，并能在多轮对话中自主调用工具（读文件、运行代码）核实问题。支持多供应商模型配置、非交互批量审查与结构化输出。

## 功能

- 多轮对话式代码审查，具备上下文记忆（`/clear` 清空、`/undo` 作废上一轮、`/save` `/load` 会话持久化）
- Agent 自主调用工具：列目录 / 读文件 / 运行 Python 代码核实问题（基于 LangGraph 的 Agent 循环 + function calling）
- 结构化审查结论：【严重/一般/建议】+ `文件:行号` + 修改建议
- 多供应商模型配置：`cra config` 交互式管理 `~/.cra/models.json`，连通性自检（`cra config test`），REPL 内 `/change model` 热切换（上下文保留）
- 非交互审查：`cra review` 多路径批量、读 stdin、`-o` Markdown 报告、`--json` 结构化输出与分级退出码
- 流式彩色输出（行级粒度：完整行即时渲染，围栏/列表整组不切段；`--no-stream` 可关）；LLM 瞬时失败自动重试（SDK 内建，认证错误快速失败）

## 架构

```
┌──────────────┐  user input / CLI args  ┌──────────────┐   model I/O   ┌────────────────┐
│    cli.py    │ ──────────────────────► │   agent.py   │ ────────────► │ langchain 1.x  │
│ chat/review/ │ ◄────────────────────── │ create_agent │ ◄──────────── │ ChatOpenAI     │
│ config /渲染 │   stream + callbacks    │ 循环/上下文   │  tool_calls   │ (OpenAI 兼容)  │
└──────┬───────┘                         └──────┬───────┘               └────────────────┘
       │ ~/.cra/models.json                     │ @tool 桥接
┌──────▼───────┐                         ┌──────────────────┐
│  config.py   │                         │ tools / registry │
│ env + 文件   │                         └──────────────────┘
└──────────────┘                           list_dir / read_file / run_python
```

Agent 循环基于 LangChain 1.x `create_agent`（底层 LangGraph），工具、提示词与流式渲染为自研封装；
分层设计与设计决策详见 [docs/Design.md](docs/Design.md)。

## 快速开始

前置：Python 3.13+、[uv](https://docs.astral.sh/uv/)、任一 OpenAI 兼容供应商的 API key（默认 DeepSeek）。

```bash
# 1. 安装依赖
uv sync

# 2. 配置（二选一）
uv run cra config          # 方式 A：交互式配置（推荐，支持多供应商）
setx SE_CodeAgent "sk-你的key"   # 方式 B：环境变量（Windows；setx 后需新开终端生效）
                                 # macOS/Linux：export SE_CodeAgent="sk-你的key"

# 3. 启动
uv run cra chat
```

裸 `cra` 命令安装（可选）：`uv tool install -e .` 之后可直接运行 `cra`（无需 `uv run` 前缀）。

首次运行时若无任何配置，会打印 `cra config` 引导并以退出码 2 结束。

## cra config：模型配置

交互式管理 `~/.cra/models.json`（用户目录，不随仓库分发）：增删供应商、管理供应商的模型清单、设置默认模型、修改 key；同名供应商/同名模型已存在时会询问是否覆盖。每次写入都会提示文件权限与备份注意事项。

```bash
uv run cra config        # 交互式配置
uv run cra config test   # 连通性自检：对每个供应商发一次 max_tokens=1 请求，报告 OK / 401 / 超时
```

配置文件格式（`default` 以第一个 `/` 分隔"供应商/模型"）：

```json
{
  "default": "deepseek/deepseek-flash",
  "providers": [
    {
      "name": "deepseek",
      "base_url": "https://api.deepseek.com",
      "api_key": "sk-...",
      "models": ["deepseek-flash", "deepseek-v4-pro"]
    }
  ]
}
```

**⚠ 文件权限与备份注意事项**：该文件以**明文**保存 API key，属本机敏感数据。请保持仅本人可读（POSIX 下写入时自动收紧为 0600；Windows 下请勿放在共享目录），备份时同样注意保密——勿上传到仓库、网盘或聊天工具。

解析优先级：`--model/--provider` 参数 > 配置文件 `default` > 环境变量。供应商条目缺 `api_key` 时回退环境变量 `SE_CodeAgent`（是否有效由认证环节暴露）。

### 环境变量（models.json 缺省时的回退）

| 环境变量 | 必填 | 默认 | 说明 |
|----------|------|------|------|
| `SE_CodeAgent` | 配置文件缺省时必填 | — | API key（仅启动校验非空） |
| `CRA_MODEL` | 否 | `deepseek-flash` | 模型名（OpenAI 兼容接口均可） |
| `CRA_BASE_URL` | 否 | `https://api.deepseek.com` | OpenAI 兼容接入点 |
| `CRA_EXEC_TIMEOUT` | 否 | `10` | 代码执行超时（秒，1~60） |
| `CRA_MAX_CONTEXT_TOKENS` | 否 | `100000` | 上下文截断阈值（tokens） |

## 使用示例

```text
$ uv run cra chat
code-review-agent 已就绪（模型：deepseek/deepseek-flash）。提出审查需求即可；输入 /help 查看命令，exit 退出。
你> 审查 src/cra/agent.py
⚙ read_file({"path": "src/cra/agent.py"})
  ↳ 完成（9xxx 字符）

【严重】src/cra/agent.py:42 — 工具调用缺少超时控制，可能永久阻塞。
修改建议：为 subprocess 调用设置 timeout 参数并捕获 TimeoutExpired。
【一般】src/cra/agent.py:57 — 异常被静默吞掉。
修改建议：将异常信息结构化回传模型。

你> 第 1 条为什么严重？
（多轮上下文延续，Agent 基于前文作答）
你> /context
上下文约 3821 tokens / 截断阈值 100000（真实 usage），余量 96179 tokens。
你> exit
再见。
```

### REPL 内置命令

| 命令 | 行为 |
|------|------|
| `/help` | 命令列表 |
| `/clear` | 清空上下文（系统提示词保留） |
| `/change model`（别名 `/model`） | 列出供应商与模型清单、序号切换；上下文保留 |
| `/save [名称]` / `/load <名称>` | 会话持久化到 `~/.cra/sessions/<名称>.json` / 恢复（整体替换） |
| `/undo` | 作废上一轮对话 |
| `/context` | 显示 token 用量与距截断阈值的余量 |
| `/export [文件]` | 将最近一轮结论导出为 Markdown（默认 `cra-export.md`） |
| `/process` | 显示最近一轮的过程明细（叙述全文/工具调用/结果预览） |
| `/confirm [on\|off]` | run_python 执行前逐次确认（默认 off） |
| `exit` 或 Ctrl+C | 退出 |

### cra review：非交互审查

```bash
uv run cra review a.py                          # 审查单文件
uv run cra review a.py b.py src/                # 多路径=批量（目录展开第一层 .py）
cat a.py | uv run cra review -                  # 从 stdin 读入
uv run cra review a.py -o report.md             # 结论写入 Markdown 报告
uv run cra review a.py --json                   # 结构化 JSON 输出（stdout）
```

退出码：`0` 审查完成；`1` 仅 `--json` 模式下按 schema 的 `severity` 判定存在【严重】问题；`2` 运行错误（参数错误、配置缺失、网络/认证失败、文件不可读等，含顶层兜底）。非 `--json` 模式的结论是自由文本，不做分级匹配；需要程序化分级判定时必须用 `--json`。

`--json` 输出 schema（stdin 输入时 `file` 为 `"<stdin>"`）：

```json
{
  "files": [
    {
      "file": "a.py",
      "findings": [
        {"severity": "严重|一般|建议", "location": "a.py:42", "issue": "…", "suggestion": "…", "uncertain": false}
      ],
      "summary": "该文件整体评价"
    }
  ],
  "summary": "整体评价"
}
```

### 全局参数（chat / review 通用）

| 参数 | 说明 |
|------|------|
| `--model <name>` / `--provider <name>` | 临时覆盖本次使用的模型/供应商（不改配置文件） |
| `--no-stream` | 关闭流式渲染，整体输出 |
| `--max-rounds <n>` | 覆盖最大工具轮次（默认 10） |
| `--ask-exec` | run_python 执行前逐次人工确认（默认自动执行；REPL 内 `/confirm on\|off` 同效）。仅限 chat；`cra review` 指定时报错退出 2 |

## Agent 可用的工具

| 工具 | 说明 | 限制 |
|------|------|------|
| `list_dir` | 查看目录结构（两层，忽略缓存目录） | — |
| `read_file` | 带行号读取文件内容 | 单次 400 行，可分段 |
| `run_python` | 运行 Python 代码/文件核实问题 | 本机直接运行、非沙箱、默认 10 秒超时 |

## 限制

- `run_python` 在本机直接执行代码（**非沙箱、限时执行**：默认 10 秒，超时强制终止整棵进程树）：请勿在审查不可信代码时滥用执行功能。子进程环境经净化后继承（剔除 `SE_CodeAgent` 与 `*API_KEY*` 型变量），但子进程仍拥有当前用户全部权限。该限制同时明示于 `cra chat --help` / `cra review --help` 与工具 description；`--ask-exec` / `/confirm` 可开启逐次执行确认。
- 审查支持任意文本型源码与配置文件；执行验证仅支持 Python（`run_python`），其他语言的结论以静态审查为准。
- `~/.cra/models.json` 含明文 API key（见上文权限与备份注意事项）；api_key 只存在于用户侧配置文件与进程内存，不进入仓库、日志与子进程环境。
- 审查意见由大模型生成，仅供参考，不能替代人工评审。

## 开发

```bash
uv sync              # 安装全部依赖（含开发依赖）
uv run pytest        # 运行测试
uv run ruff check .  # 代码规范检查
```

项目结构与设计文档：[docs/Design.md](docs/Design.md)。

## 效果截图

（占位小节——截图由作者在提交前补充：单轮审查 / 多轮追问 / 边界样例表现，可对照
`tests/fixtures/buggy_samples/` 中的样例演示超时强杀与非 Python 文件提示。）

## Roadmap

- [x] 审查结果导出 Markdown 报告（`cra review -o` / `/export`）
- [x] 批量审查多文件（`cra review` 多路径）
- [x] 流式过程体验优化（行级流式 + 句级早发、过程/正文分层、`/process` 过程重放；设计见 docs/Design.md D15/D16）

---

本项目由 AI 编码会话辅助开发；人机分工与开发过程记录见 [docs/Design.md](docs/Design.md) §8。
