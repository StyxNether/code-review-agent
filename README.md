# code-review-agent · 代码审查助手

基于 LLM 的命令行代码审查工具。给它一段/一个/一组代码文件（任意文本型源码、脚本与配置文件，不限编程语言），它以代码审查员身份输出分级问题清单，并能在多轮对话中自主调用工具（读文件、运行代码）核实问题。支持多供应商模型配置、非交互批量审查与结构化输出。

## 功能

- 多轮对话式代码审查，具备上下文记忆（`/clear` 清空、`/undo` 作废上一轮、`/save` `/load` `/delete` 会话管理、退出自动保留会话 + `/resume` 恢复）
- Agent 自主调用工具：列目录 / 读文件 / 运行 Python 代码核实问题（基于 LangGraph 的 Agent 循环 + function calling）
- 结构化审查结论：【严重/一般/建议】+ `文件:行号` + 修改建议
- 多供应商模型配置：`cra config` 交互式管理 `~/.cra/models.json`（内置 DeepSeek/Kimi/OpenAI/百炼/Z.ai/BigModel 等 8 家预设，选编号即自动填官方接入点与模型清单），key 可选明文或环境变量存储，连通性自检（`cra config test`），REPL 内 `/change model` 热切换（上下文保留）与 `/setting` 设置菜单
- 首次使用引导：新机器无任何配置时 `cra` 会询问"是否现在配置"，确认后按 添加供应商 → key → 设默认模型 引导完成并直接进入对话
- 非交互审查：`cra review` 多路径批量、读 stdin、`-o` Markdown 报告、`--json` 结构化输出与分级退出码
- 流式彩色输出（行级粒度：完整行即时渲染，围栏/列表整组不切段；`--no-stream` 可关）；LLM 瞬时失败自动重试（SDK 内建，认证错误快速失败）
- 一键卸载：`cra uninstall` 多道交互确认后分级清理程序、垫片、PATH 条目、用户数据与本程序设置的环境变量

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
setx CRA_DEEPSEEK_API_KEY "你的key"   # 方式 B：环境变量（Windows；setx 后需新开终端生效）
                                      # macOS/Linux：export CRA_DEEPSEEK_API_KEY="你的key"

# 3. 启动（`cra` 省略子命令默认进入对话，等价于 `uv run cra chat`）
uv run cra
```

裸 `cra` 命令安装（可选）：`uv tool install -e .` 之后可直接运行 `cra`（无需 `uv run` 前缀）；省略子命令时默认进入对话式审查（裸形式不带参数，需传选项时写作 `cra chat --no-stream` 等）。

首次运行（新机器）：若无任何配置，`cra` 会询问"是否现在配置"——回车确认后按 **添加供应商（可选预设）→ 填 API key → 设默认模型** 引导完成，随即进入对话；拒绝则打印 `cra config` 指引并以退出码 2 结束。`cra review` 为非交互模式，无配置时直接打印指引退出 2。

## cra config：模型配置

交互式管理 `~/.cra/models.json`（用户目录，不随仓库分发）：增删供应商、管理供应商的模型清单、设置默认模型、修改 key；同名供应商/同名模型已存在时会询问是否覆盖。每次写入都会提示文件权限与备份注意事项。

```bash
uv run cra config        # 交互式配置
uv run cra config test   # 连通性自检：对每个供应商发一次 max_tokens=1 请求，报告 OK / 401 / 超时
```

**供应商预设**：添加供应商时提供编号选择——1 自定义 + 8 家预设（DeepSeek、Kimi (Moonshot)、OpenAI、阿里云百炼、Z.ai Coding Plan / Z.ai API、BigModel Coding Plan / BigModel API）。选预设自动填入官方 base_url 与模型清单，**预填项均可修改**（base_url 回车用官方接入点或输入自定义；模型清单可答 n 自行输入）。

**key 存储选择**：填入 key 后可选——1) 明文写入 models.json（注意文件权限与备份）；2) 环境变量（变量名统一为 `CRA_供应商名大写_API_KEY`，如 DeepSeek → `CRA_DEEPSEEK_API_KEY`，自定义供应商按"供应商名大写、空格与非字母数字转 `_`"生成）。**若该变量已存在且不是本程序设置的**（例如您此前已配置过），会先询问是否覆盖（默认不覆盖，拒绝则 key 改存明文、原值不动）；本程序此前设置过的变量（登记在案）直接更新、不再重复询问。本程序写入的变量会登记到 `~/.cra/env-vars.json`，供卸载时精确清理。Windows 经 `setx` 自动写入并即时生效（key 不回显）；macOS/Linux 打印 `export` 指令由用户执行。选环境变量时 models.json 不落明文，已有明文一并移除。

配置文件格式（`default` 以第一个 `/` 分隔"供应商/模型"）：

```json
{
  "default": "DeepSeek/deepseek-flash",
  "providers": [
    {
      "name": "DeepSeek",
      "base_url": "https://api.deepseek.com",
      "models": ["deepseek-flash", "deepseek-v4-pro"]
    }
  ]
}
```

**⚠ 文件权限与备份注意事项**：选择明文存储时该文件以**明文**保存 API key，属本机敏感数据。请保持仅本人可读（POSIX 下写入时自动收紧为 0600；Windows 下请勿放在共享目录），备份时同样注意保密——勿上传到仓库、网盘或聊天工具。

解析优先级：`--model/--provider` 参数 > 配置文件 `default` > 环境变量。供应商条目缺 `api_key` 时按序回退环境变量：`CRA_供应商名大写_API_KEY`（如 DeepSeek 为 `CRA_DEEPSEEK_API_KEY`）→ `SE_CodeAgent`（历史变量，最后回退；key 是否有效由认证环节暴露）。

### 环境变量（models.json 缺省时的回退）

| 环境变量 | 必填 | 默认 | 说明 |
|----------|------|------|------|
| `CRA_DEEPSEEK_API_KEY` | 配置文件缺省时必填 | — | 默认供应商（DeepSeek）的 API key（仅启动校验非空） |
| `SE_CodeAgent` | 否（历史变量） | — | 旧版 key 变量名，仍作最后回退 |
| `CRA_MODEL` | 否 | `deepseek-flash` | 模型名（OpenAI 兼容接口均可） |
| `CRA_BASE_URL` | 否 | `https://api.deepseek.com` | OpenAI 兼容接入点 |
| `CRA_EXEC_TIMEOUT` | 否 | `10` | 代码执行超时（秒，1~60） |
| `CRA_MAX_CONTEXT_TOKENS` | 否 | `100000` | 上下文截断阈值（tokens） |

各供应商 key 的环境变量名统一为 `CRA_供应商名大写_API_KEY`（如 `CRA_DEEPSEEK_API_KEY`、`CRA_MOONSHOT_API_KEY`、`CRA_OPENAI_API_KEY`、`CRA_DASHSCOPE_API_KEY`、`CRA_ZAI_API_KEY`、`CRA_ZHIPUAI_API_KEY`）——`cra config` 选环境变量存储时自动生成；回退顺序见上文"解析优先级"。

## 使用示例

```text
$ uv run cra
code-review-agent 已就绪（模型：DeepSeek/deepseek-flash）。提出审查需求即可；输入 /help 查看命令，exit 或 /exit 退出。
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
| `/change model`（别名 `/model`） | 列出供应商与模型清单（标注"当前使用中"）、序号切换；上下文保留 |
| `/setting` | 设置菜单（数字选择）：切模型 / 配置模型列表 / 保存 / 加载 / 执行确认开关 / 删除会话；0 或回车返回对话 |
| `/save [名称]` | 会话持久化到 `~/.cra/sessions/<名称>.json`（省略名称自动以时间戳命名） |
| `/load [名称]` | 恢复会话（整体替换当前会话）；省略名称时列出会话数字选择（回车取消）；加载成功后自动打印全部会话历史 |
| `/delete` | 删除会话：列表数字选择 + y/N 确认（`_last` 同样可删） |
| `/resume` | 恢复最近一次自动保留的会话（`_last`） |
| `/undo` | 作废上一轮对话 |
| `/context` | 显示 token 用量与距截断阈值的余量 |
| `/export [文件]` | 将最近一轮结论导出为 Markdown（默认 `cra-export.md`） |
| `/process` | 显示最近一轮的过程明细（叙述全文/工具调用/结果预览） |
| `/confirm [on\|off]` | run_python 执行前逐次确认（默认 off） |
| `exit` 或 `/exit`（或 Ctrl+C） | 退出；退出时自动保留会话（覆盖写入）到 `~/.cra/sessions/_last.json`，下次启动提示 `/resume` 可恢复 |

### cra review：非交互审查

```bash
uv run cra review a.py                          # 审查单文件
uv run cra review a.py b.py src/                # 多路径=批量（目录展开第一层文本文件，跳过隐藏与二进制文件）
cat a.py | uv run cra review -                  # 从 stdin 读入
uv run cra review a.py -o report.md             # 结论写入 Markdown 报告
uv run cra review a.py --json                   # 结构化 JSON 输出（stdout）
```

退出码：`0` 审查完成；`1` 仅 `--json` 模式下按 schema 的 `severity` 判定存在【严重】问题；`2` 运行错误（参数错误、配置缺失、网络/认证失败、文件不可读等，含顶层兜底）。非 `--json` 模式的结论是自由文本，不做分级匹配；需要程序化分级判定时必须用 `--json`。

> 注意：`cra review` 为非交互模式，无法使用 `--ask-exec`/`/confirm` 逐次确认执行；审查不可信来源的代码时，Agent 仍可能按审查流程自主运行 run_python 核实问题（本机直接运行、非沙箱）。如需执行前逐次确认，请改用 `cra chat` 并开启 `/confirm on`。

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

### cra uninstall：一键卸载

```bash
cra uninstall
```

定位：**只删除项目文件夹之外、且由本程序产生的东西**——克隆的项目文件夹本身请自行删除（如 `rm -rf code-review-agent`），两者合起来即是完整退场。

三道交互确认（第一道默认 **N** 防误触发，后两道默认 **Y** 满足"删干净"，Ctrl+C 一律保留；环境变量一问仅在存在已登记变量时出现）后分级清理：

1. **清单确认**：列出将删除内容（uv 工具 `code-review-agent` 与 cra 垫片、PATH 条目、用户数据、本程序设置的环境变量清单），回车或 `n` 取消、不删除任何内容；
2. **用户数据**：删除 `~/.cra/`（models.json 含 API key、sessions/ 会话存档、环境变量登记文件），答 `n` 保留；
3. **环境变量**：仅清理 `~/.cra/env-vars.json` 中**登记在案**的变量（本程序 `setx` 写入过的），逐个列出明示——您已有的其它变量（例如您此前已自行设置的 `CRA_DEEPSEEK_API_KEY`）绝不触碰；待删清单在删除 `~/.cra/` **之前**已读出（登记文件就在其中）；Windows 经 PowerShell 从用户环境移除（新开终端立即可见），macOS/Linux 打印待删的 export 行（不自动改 rc）；登记文件损坏时保守跳过；
4. **后台清理**：生成等待本进程退出后执行 `uv tool uninstall` 的分离后台任务（运行中的解释器文件被锁，须父进程退出后才能删净），完成后若已无其他 uv 工具，再从用户 PATH 移除 `~/.local/bin` 条目（Windows 自动；macOS/Linux 打印待删的 shell 配置行指令、不自动改 rc）；
5. 打印"程序将在退出后完成卸载"后正常退出。

边界：`uv tool list` 中无本工具（如以 `uv run` 方式使用）时跳过工具卸载并说明；PATH 条目被其他 uv 工具共享时保留。中途任何一步拒绝都无副作用。

### 全局参数（chat / review 通用）

参数跟在子命令之后（如 `cra chat --no-stream`；裸 `cra` 形式不带参数）。

| 参数 | 说明 |
|------|------|
| `--model <name>` / `--provider <name>` | 临时覆盖本次使用的模型/供应商（不改配置文件） |
| `--no-stream` | 关闭流式渲染，整体输出（review 恒为非流式，此参数仅为参数面一致保留） |
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
- key 存储二选一（`cra config` 内选择）：环境变量（推荐，不落盘）或 `~/.cra/models.json` 明文（见上文权限与备份注意事项）；api_key 只存在于用户侧（配置文件/环境变量）与进程内存，不进入仓库、日志与子进程环境。
- 审查意见由大模型生成，仅供参考，不能替代人工评审。

## 开发

```bash
uv sync              # 安装全部依赖（含开发依赖）
uv run pytest        # 运行测试
uv run ruff check .  # 代码规范检查
```

项目结构与设计文档：[docs/Design.md](docs/Design.md)。

## 效果演示

可对照 `tests/fixtures/buggy_samples/` 中的样例演示：单轮审查（syntax_error.py）、
多轮追问、执行超时强杀（infinite_loop.py）、非 Python 文件提示（note.txt）与二进制
文件兜底（binary.bin）；`cra review --json` 可观察结构化输出与分级退出码。

## Roadmap

- [x] 审查结果导出 Markdown 报告（`cra review -o` / `/export`）
- [x] 批量审查多文件（`cra review` 多路径）
- [x] 流式过程体验优化（行级流式 + 句级早发、过程/正文分层、`/process` 过程重放；设计见 docs/Design.md D15/D16）
- [x] CLI 易用性（裸 `cra` 直接对话、`/setting` 设置菜单、`/exit`、会话自动保留 + `/resume`）
- [x] 会话与配置便利性（首次配置引导、8 家供应商预设、key 环境变量存储、`/load` 列表选择与历史打印、`/delete`、`cra uninstall` 一键卸载）
