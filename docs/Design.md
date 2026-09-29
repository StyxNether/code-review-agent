# Design.md — 代码审查助手 code-review-agent

《软件工程与项目实践》第 1 次作业 · 提交截止 2026-10-07 24:00（作业二将复用本项目成果，见 §2.2 与 §10）
平台环境：Windows + Git Bash（PowerShell/CMD 兼容）。

## 1. 项目概述

`cra` 是一个基于 LLM 的命令行代码审查助手：给定一段/一个/一组代码文件（任意文本型源码、脚本与配置文件，Python 为主），它以代码审查员身份给出问题清单（分级、定位、修改建议），并能在多轮对话中自主调用工具（读文件、运行代码）核实问题。支持多供应商模型配置与脚本化调用。

- 交付物：GitHub 私有仓库 + README.md / Design.md +（可选，用户自理）1 分钟演示视频；备份提交"学号姓名.rar"（<200M）至课程"001Homework1"。
- 评分权重：功能完整性 40%（核心可用、边界情况）· Agent 架构 30%（设计模式、架构清晰度）· 代码质量 20% · 文档 10%。

## 2. 范围

### 2.1 作业一范围

| 级别 | 内容 |
|------|------|
| Must | 审查代码（任意文本型源码，Python 为主）；文件读取工具（列目录 + 读单文件）；代码执行工具（subprocess 限时 10s，本机运行非沙箱，限制明示）；多轮上下文记忆；指数退避重试；CLI（彩色流式输出，入口 `cra`）；pytest 测试集；多供应商模型配置（`~/.cra/models.json` + 配置模式）；非交互审查模式 |
| Should | 审查结果导出 Markdown 报告（`cra review -o` / `/export`）；批量审查多文件（`cra review` 多路径） |
| Won't | Web 界面；多语言**执行**（run_python 仅运行 Python，任意语言源码的审查不受限）；Docker 沙箱；联网搜索工具；LLM 调用费用核算（只报 tokens） |

### 2.2 与作业二的接缝（只留缝，不提前实现）

作业二为 Simple Workflow MVP（2–3 个 Agent 协作、消息传递/状态共享、顺序流程控制，推荐条件分支/循环/并行），可复用本项目 Agent 作为 Workflow 节点。为此本设计预留三个接缝，在 M4/M5 落实：

1. **Agent 即组件**：`Agent.run(input, callbacks) -> str` 是可编程调用接口，REPL/非交互模式只是它的两个外壳；作业二将其包装为工作流节点。
2. **结构化输出**：`cra review --json` 按固定 JSON schema 输出分级问题清单，供下游 Agent 程序化消费（作业二"修复 Agent"）。
3. **编排层就位**：单 Agent 循环基于 LangChain `create_agent`（底层 LangGraph）；作业二的工作流直接用 LangGraph `StateGraph` 组合节点（顺序/分支/并行/检查点为框架原语），无需二次迁移。

## 3. 总体架构

```
┌──────────────┐  user input / CLI args   ┌──────────────┐   model I/O   ┌────────────────┐
│    cli.py    │ ───────────────────────► │   agent.py   │ ────────────► │ langchain 1.x  │
│ chat/review/ │ ◄─────────────────────── │ create_agent │ ◄──────────── │ ChatOpenAI     │
│ config /渲染 │  stream + callbacks      │ 循环/上下文   │  tool_calls   │ (OpenAI 兼容)  │
└──────────────┘                          └──────┬───────┘               └────────────────┘
       │                                        │ @tool
       │ ~/.cra/models.json                     ▼
┌──────▼───────┐                        ┌──────────────┐
│  config.py   │                        │ tools/ 注册表 │
│ env + 文件   │                        │ fs / exec_py │
└──────────────┘                        └──────────────┘
```

模块划分（src 布局，包目录固定为 `src/cra`）：

```
src/cra/
  cli.py        # argparse 子命令（chat/review/config）+ REPL + rich 流式彩色渲染
  agent.py      # create_agent 封装：系统提示词组装、工具绑定、最大轮次、上下文截断、回调适配
  llm.py        # ChatOpenAI 封装：客户端构建（base_url/api_key/温度/超时/重试/认证快速失败）
  config.py     # 配置层：~/.cra/models.json 与环境变量的合并读取（唯一出入口）
  prompts.py    # 系统提示词与审查输出格式约定
  tools/
    __init__.py # 工具注册表：名称 → (JSON schema, handler)，@tool 包装入口
    fs.py       # list_dir、read_file
    exec_py.py  # run_python（subprocess 限时执行）
  sessions.py   # 会话持久化（/save /load，M5 新增）
scripts/
  smoke_tools.py   # 冒烟测试（入库留存）：真实网络验证 function calling 通路
tests/
  fixtures/buggy_samples/   # 预埋已知问题的样例（黄金回归用例）
```

### 3.1 Agent 设计

"输入 → 推理 → 工具调用 → 输出"基于 LangChain 1.x `langchain.agents.create_agent` 实现（底层 LangGraph）：

1. 用户输入追加进会话消息；系统提示词常驻（角色 + 输出格式 + 工具使用策略，见 §3.3）。
2. 模型客户端为 `ChatOpenAI`（OpenAI 兼容，`base_url`/`api_key` 来自配置层），`temperature=0.2`（输出稳定，黄金用例可复现），请求级超时 60s。
3. 工具经 `@tool` 包装现有纯函数 handler；工具失败仍以"错误：…"结构化文本返回给模型（模型可自行调整策略），不向上抛出。模型返回未注册工具或参数不符时同样以结构化错误回传。
4. **最大工具轮次 10**：达到上限后注入指令并以不带工具的调用强制收尾，保证循环必然终止。
5. **上下文记忆** = 会话消息跨轮持续存在；截断语义与既有一致——以真实 usage 超过阈值（默认 100000 tokens）触发，按**完整轮次**原子丢弃、`tool_calls`/`tool` 配对不拆散、系统提示词永留（迁移后经 LangChain middleware 或等价钩子实现，语义由回归测试锁定）。
6. **流式渲染**：最终回答的增量经回调链交给 cli 的渲染管线（逐段 Markdown + 分级标记着色）；工具轮不渲染文本增量。统一采用流式请求（轮次类型无法预判；usage 经响应元数据取得，缺失时截断不触发——安全降级）。
7. `Agent.run(input, callbacks) -> str` 为可编程接口（§2.2 接缝 1）；任何异常（Ctrl+C 或请求失败）回滚本轮消息到合法前缀后原样上抛。

### 3.2 工具 schema

| 工具 | 参数 | 行为与限制 |
|------|------|-----------|
| `list_dir` | `path` | 返回两层目录树；忽略 `.venv`、`__pycache__`、`.git`、`.pytest_cache`、`.ruff_cache` 与点开头的隐藏目录 |
| `read_file` | `path`, `start?`, `end?` | 带行号返回内容；单次上限 **400 行**，超出提示用 start/end 分段；`start`/`end` 为 1-based 闭区间，非法区间返回结构化错误；二进制/解码失败返回结构化错误信息（回传模型） |
| `run_python` | `code` 或 `path`（二选一）, `timeout?` | subprocess 运行，工作目录=进程启动目录；默认 10s（`CRA_EXEC_TIMEOUT` 可调），Windows 下 `taskkill /T /F` 强杀整棵进程树；返回 stdout/stderr/退出码；stdout/stderr 各截断至 **2000 字符**；**本机直接运行，非沙箱**；执行确认开关见 §7.4 |

通用规则：三工具的相对路径一律相对**进程启动目录（cwd）**解析；所有文件读取与子进程输出捕获显式 `encoding='utf-8', errors='replace'`，子进程环境另设 `PYTHONIOENCODING=utf-8`。

### 3.3 系统提示词要点

角色为资深代码审查员（Python 为主深耕，任意语言的源码、脚本与配置文件均可审查；执行验证仅 Python，其他语言静态审查并标注"不确定"）；输出固定结构：【严重/一般/建议】分级 + `文件:行号` + 问题 + 修改建议（修改建议独立成行、紧跟问题描述）；先调用工具核实再下结论；不确定的问题明确标注"不确定"，禁止编造行号或代码。默认输出格式为**提示词约定**，不做程序化校验；`--json` 模式（§7.1）例外：按 §7.5 的 schema 程序化输出与校验。

### 3.4 代码设计规范落点

风格层由 ruff 强制（§6）；设计层按以下原则执行，各里程碑收尾按 §8.2 Checklist 核对：

| 原则 | 本项目落点 |
|------|-----------|
| 单一职责 | cli 只管交互与渲染；agent 只管循环与状态；llm 只管模型客户端与重试；tools 只管工具执行；config 是配置唯一出入口 |
| 依赖与模块化 | 依赖方向单向：cli → agent → {llm, tools}；tools/llm 不反向 import agent/cli。框架类型（LangChain 消息/流对象）不越过 agent 层向上暴露 |
| 数据封装 | 会话消息结构与截断规则只存在于 agent.py；工具返回统一为字符串（错误也是字符串） |
| 异常处理 | 按 §5：异常要么被结构化回传模型，要么给出用户可读报错，不允许静默吞掉 |
| 性能 | 流式输出保证感知性能；工具调用有超时；上下文有截断阈值 |
| 安全 | key 只在内存流转、不进仓库（用户目录配置文件除外，见 §7.4）；run_python 限时 + 截断 + 限制明示（第3讲 Gab 教训：规范一丢，漏洞就来） |

## 4. 关键决策记录（ADR）

决策按日期记录，被取代的决策保留条目并标注状态——决策演化本身是项目过程的一部分。

| # | 决策 | 状态 | 理由与后果 |
|---|------|------|-----------|
| D1 | 手写 ReAct 循环（消息列表 + function calling + 循环分发），不引入框架 | 已取代（→D8） | 作业一 M0–M3 以原生 API 路线实现并验收；吃透 Agent 原理、架构可讲可测。被取代原因见 D8 |
| D2 | LLM 用 DeepSeek，OpenAI 兼容接口 | 采纳 | 国内直连、成本低；OpenAI 兼容接入点可替换服务商 |
| D3 | 仅 CLI，入口 `cra`，不做 Web 界面 | 采纳 | 作业允许二选一；聚焦 Agent 核心能力 |
| D4 | run_python 用 subprocess + 限时强杀，不做沙箱 | 采纳 | 作业范围权衡；风险用"限制明示 + 超时强杀 + 输出截断 + 可选执行确认"控制 |
| D5 | 统一流式请求（轮次类型无法预判），工具轮不渲染增量 | 采纳 | tool_calls 分片组装完整性等价；用户可见行为为"仅最终回答流式"；不使用 rich.Live（Windows 控制台重绘不可靠） |
| D6 | 上下文截断按完整轮次原子丢弃，系统提示词永留 | 采纳 | 拆散 tool_calls/tool 配对会导致 API 400；纯函数实现、回归锁定 |
| D7 | 审查输出格式为提示词约定，不做程序化校验 | 采纳（由 D11 扩展） | 格式断言测试脆弱；`--json` 是显式例外（schema 校验） |
| D8 | Agent 循环迁移至 LangChain 1.x `create_agent`（底层 LangGraph） | 采纳（2026-09-29，用户解除 D1） | 作业二要求 2–3 Agent 工作流并复用本项目：LangGraph 的节点/边/状态原语直接支撑编排与可扩展性（作业二评分项）；`create_agent` 为官方现行推荐 API；工具层/提示词/渲染/测试策略全部可复用。备选 AutoGen/CrewAI（多智能体叙事错位）、Dify（平台非库，违反仅 CLI）已排除；后果：接受框架版本漂移风险（§9），以 uv.lock 锁定管理 |
| D9 | 多供应商模型配置：`~/.cra/models.json`（用户目录，仓库外）+ 环境变量回退 | 采纳 | 用户要求命令行配置供应商/密钥/模型；key 存用户目录可同时满足"配置体验"与"仓库无明文密钥"门禁；JSON 程序化读写优于 Markdown |
| D10 | 非交互审查模式 `cra review`（含批量/报告导出/退出码） | 采纳 | CLI 审查工具的标配形态；一并落地 Should 两条；是作业二"Agent 即节点"的雏形 |
| D11 | run_python 默认自动执行，`--ask-exec`/`/confirm` 可开启逐次确认 | 采纳 | 审查中执行验证是高频动作，默认打断伤体验；安全叙事由文档明示 + 可选开关表达 |
| D12 | 文档以最终形态维护（不设版本号），决策历史由本表与 §8 PSP 附录承载 | 采纳 | 无需保留旧版本；决策演化靠 ADR 状态列追溯 |

## 5. 错误处理与重试

行为契约（实现映射到 LangChain SDK 能力，由回归测试锁定）：

- 启动校验：无任何可用配置（配置文件与环境变量皆缺）时打印 `cra config` 设置指引后退出；**不校验 key 格式**（无效 key 由 API 认证环节暴露）。
- LLM 调用：超时/限流/连接错误 → 指数退避重试（1s/2s/4s，最多 3 次；请求级超时 60s）；仍失败给出友好报错与排查建议；**认证类错误（401/403）快速失败、不重试**，输出 key 排查指引（跨平台：Windows setx / macOS-Linux export）。
- 工具执行：handler 内所有异常捕获 → 结构化错误文本作为工具结果回传模型，REPL 不崩溃。
- `run_python`：超时强杀进程树；stdout/stderr 截断防刷屏。
- `Ctrl+C`：中断当前生成——任何中断必须把会话消息回滚到本轮开始前的合法前缀并提示该轮作废；输入 `exit` 或再次 `Ctrl+C` 退出 REPL。

## 6. 工程规格

- 包管理：uv（≥0.12），`requires-python = ">=3.13"`，`.python-version` 固定 `3.13`；包目录 `src/cra`，hatchling 打包配置 `packages = ["src/cra"]`。
- 运行依赖：`langchain>=1.0`、`langchain-openai>=1.0`、`rich`（M4 迁移时以 uv.lock 锁定）。开发依赖：`pytest`、`ruff`。新增依赖须先征得用户同意。
- 入口与 CLI 面：`[project.scripts]` → `cra = "cra.cli:main"`；子命令 `chat` / `review` / `config`（§7）。全局安装：`uv tool install -e .` 后可直接 `cra`（README 载明）。
- 配置（环境变量，作为 `~/.cra/models.json` 不存在时的回退，统一在 `config.py` 读取）：

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `SE_CodeAgent` | 回退必填 | — | DeepSeek API key（配置文件缺省时的回退） |
| `CRA_MODEL` | 否 | `deepseek-flash` | 模型名回退 |
| `CRA_BASE_URL` | 否 | `https://api.deepseek.com` | 接入点回退 |
| `CRA_EXEC_TIMEOUT` | 否 | `10` | run_python 超时（秒） |
| `CRA_MAX_CONTEXT_TOKENS` | 否 | `100000` | 会话截断阈值（tokens） |

- ruff：默认规则 + `line-length = 100`；`tests/fixtures` 豁免（预埋样例属数据非业务代码）。
- pytest `testpaths = ["tests"]`；**测试内禁止真实网络调用（LLM 一律 mock），真实网络仅限 scripts/smoke_tools.py**。
- `.gitignore`：`.venv/`、`__pycache__/`、`.pytest_cache/`、`.ruff_cache/`、`dist/`、`.env`、`*.pyc`、`AGENTS.md`、`docs/PROGRESS.md`。开发辅助文档（AGENTS.md、PROGRESS.md）仅本地维护、不随仓库分发；仓库内交付文档为 README.md 与 Design.md。
- 密钥边界：api_key 只允许出现在用户目录 `~/.cra/models.json`（§7.4）与进程内存；**仓库内任何文件、日志、测试样例、提交记录不得出现真实密钥**。

## 7. 接口规格

### 7.1 CLI 子命令

```
cra chat   [选项]                  # 进入对话式审查 REPL
cra review <path...> | -  [选项]   # 非交互一次性审查（- 读 stdin；多路径=批量）
cra config [test]                  # 交互式配置模式；test 为连通性自检
cra --version                      # 版本号
```

`cra review` 选项：`-o <file>` 将审查结论写入 Markdown 报告；`--json` 以 §7.5 schema 输出结构化结果（stdout）；其余选项与 chat 一致。

### 7.2 全局参数（chat / review 通用）

| 参数 | 说明 |
|------|------|
| `--model <name>` / `--provider <name>` | 临时覆盖本次使用的模型/供应商（不改配置文件） |
| `--no-stream` | 关闭流式渲染，整体输出 |
| `--max-rounds <n>` | 覆盖最大工具轮次（默认 10） |
| `--ask-exec` | run_python 执行前逐次人工确认（默认自动执行；REPL 内 `/confirm on|off` 同效） |

### 7.3 REPL 内置命令

| 命令 | 行为 |
|------|------|
| `/help` | 命令列表 |
| `/clear` | 会话重置为仅含系统提示词（角色与格式契约保留） |
| `/change model`（别名 `/model`） | 列出供应商及其模型清单，按序号选择；切换后上下文保留并打印提示 |
| `/save [名称]` / `/load [名称]` | 会话持久化到 `~/.cra/sessions/` / 从文件恢复 |
| `/undo` | 作废上一轮对话（回滚到上一轮开始前的状态） |
| `/context` | 显示当前会话 token 用量与距截断阈值的余量 |
| `/export [文件]` | 将最近一轮审查结论导出为 Markdown |
| `/confirm [on|off]` | run_python 执行确认开关（默认 off） |
| `exit`（或 Ctrl+C 两次） | 退出 |
| `/` 以外的其余 `/xxx` | 按普通文本发送给模型 |

### 7.4 配置文件 `~/.cra/models.json`

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

- 位置固定在**用户目录**（Windows 为 `%USERPROFILE%\.cra\`），不在仓库内、不随仓库分发；`cra config` 负责创建与编辑（交互式：增删供应商/模型、设默认、改 key；**同名供应商 + 同名模型**已存在时询问是否覆盖）。
- 解析优先级：`--model/--provider` 参数 > 配置文件 `default` > 环境变量回退（`SE_CodeAgent`/`CRA_MODEL`/`CRA_BASE_URL`）。两者皆缺时打印 `cra config` 引导后退出。
- `cra config test`：对每个供应商用当前 key 发一次 1-token 请求，报告 OK / 401（key 无效）/ 超时（网络不可达）。
- 安全边界：该文件含明文 key，属用户本机数据的已知取舍（D9）；文档提示文件权限与备份注意事项，仓库侧由 .gitignore 与密钥门禁保证永不入库。

### 7.5 退出码与结构化输出

| 退出码 | 含义（`cra review`） |
|--------|---------------------|
| `0` | 审查完成，无【严重】问题 |
| `1` | 审查完成，存在【严重】问题 |
| `2` | 运行错误（配置缺失、网络/认证失败、文件不可读等） |

`--json` 输出 schema（最终结论的结构化形态，供脚本与作业二下游 Agent 消费）：

```json
{
  "file": "a.py",
  "findings": [
    {"severity": "严重|一般|建议", "location": "a.py:42", "issue": "…", "suggestion": "…", "uncertain": false}
  ],
  "summary": "整体评价一两句"
}
```

`cra chat` 正常退出码 0；配置缺失退出码 2。

## 8. PSP 附录（轻量诚实版）

**开发过程定位（第3讲）**：本计划为**增量开发**——每个里程碑交付一个可用增量，用户验收即持续反馈，回归测试保护既有增量，验收意见回灌下一里程碑。代码审查按三级落地：**静态审查**（ruff + pytest）、**动态审查**（验收运行检查）、**代码走查**（验收 + 会话收尾总结，范围 = 该里程碑 diff）、**结果记录**（PROGRESS.md 验收记录）；§8.2 Checklist 是走查门禁。

### 8.1 估算表

| 项 | 估算 | 实际 |
|----|------|------|
| 总规模 | ~1000 LOC（含测试，git 统计为准） | 3273 LOC（截至 M3 收尾，src 1240 + tests 1857 + scripts 176）；M4/M5 完成后重估 |
| 人工投入 | 设计评审 0.5h + 里程碑验收 0.25h×n + 收尾 0.5h | M0–M2 验收实测共 22 分钟；M3 及后续由用户在 PROGRESS.md 耗时表记录 |
| AI 生成 | 全部实现代码与测试初稿 | 一致：AI 编码会话生成，人工评审设计、逐里程碑验收 |
| 日历时间 | 2–3 天（4 个编码会话） | 1 天完成 M0–M3 编码与验收 |
| M4 框架迁移 | 1 个编码会话；净变化 ≈ −200 LOC（框架替代手写循环）+ 测试重写 ~400 行 | （待填） |
| M5 配置与 CLI 面 | 1–2 个编码会话；净增 ~800–1200 LOC（含测试） | （待填） |

### 8.2 代码评审 Checklist（M3 收尾门禁执行记录；M4/M5 收尾沿用本清单）

- [x] 全仓库 grep 无明文密钥；key 读取点唯一（config 层） ✅（M3 执行；M5 引入配置文件后门禁改为"仓库内无明文密钥"，`~/.cra/` 为用户数据不在仓库）
- [x] 无裸 `except:`；所有工具异常均回传模型而非静默吞掉 ✅
- [x] `run_python` 有超时强杀与输出截断 ✅
- [x] Agent 循环有最大工具轮次（10）限制 ✅
- [x] 消息截断按完整轮次原子丢弃，配对不拆散，系统提示词保留 ✅
- [x] `cra chat --help` 与 README 含"非沙箱、限时执行"限制声明 ✅
- [x] README 每条命令逐条实测可复制执行 ✅
- [x] pytest 覆盖：工具 handler、消息组装、截断逻辑、CLI 参数解析 ✅（152 项）
- [x] ruff 全绿；公开函数有类型标注；命名自解释 ✅
- [x] 模块依赖方向符合 §3.4（无反向 / 越层 import） ✅
- [x] `git status` 干净：无 venv/缓存/密钥/开发辅助文档被跟踪 ✅
- [x] Design.md 与实现一致（模型、工具名、模块、入口、配置项、CLI 面） ✅

### 8.3 复盘、缺陷记录与 PIP

- **缺陷记录**（M0–M3 验收与走查共 19 项，全部闭环、无遗留）：
  - M0 验收 1 项：提示词角色未定义【一般】→ M1 prompts.py 落实。
  - M1 验收 1 项：输出文字堆叠不便阅读【建议】→ 修复轮 1（空行/加粗）。
  - M1 走查（cra 复审两轮）13 项：中断回滚索引失位致孤儿 tool_calls【严重】、空回答静默入库【一般】→ 修复轮 1；标记加粗漏标、围栏奇偶误判、工具失败判定、rich markup 注入、REPL 意外异常兜底、llm 异常包装扩大【一般 6】及显式化小项【建议 5】→ 修复轮 2。
  - M2 编码期 1 项：read_file 空文件误报区间错误【一般】→ 现场修复。
  - M2 验收 2 项：审查语言边界设置错误【一般】、修改建议格式与着色【建议】→ 修复轮（多语言放开、标记切片渲染）。
  - M2 修复轮自测回归 1 项：CommonMark flanking 规则致标记加粗失效【一般】→ 渲染改标记切片。
  - M3 挂账处置：finish_reason 透传、print/console 统一、指引跨平台化、_run_chat 可测性已落地；stream_options 不做降级重试、截断 O(n²) 维持不改（评估记录见 PROGRESS.md）。
- 估算偏差：（人工/规模/日历——规模见 §8.1，其余由用户结合耗时表填写）
- 过程问题：
- PIP（结对项目将在本项目上增量开发与重构——的改进项）：

## 9. 风险与限制

- **框架版本漂移**（D8 后新增）：LangChain 迭代快，以 uv.lock 锁定版本；升级属重构，须回归全绿。
- `run_python` 非沙箱：仅用于本地可信代码的审查场景，文档多处明示；子进程默认继承父环境变量（含 key），属已知边界，如需收紧可传入净化后的 env。
- key 明文存于用户目录配置文件：用户本机数据的已知取舍（D9）；仓库侧以 .gitignore 与密钥门禁兜底，文档提示文件权限。
- 审查质量依赖模型能力：对专有框架/最新库版本可能误报，提示词已要求标注"不确定"。
- 上下文超长仅做轮次截断不做摘要；usage 缺失（服务端不支持）时截断不触发（安全降级）。

## 10. Roadmap 与 Backlog

**已排期**（开工提示词见 PROGRESS.md）：

- **M4 框架迁移**：langchain/langchain-openai 接入；llm.py → ChatOpenAI 封装；agent.py → create_agent 封装；工具 @tool 包装；行为不变（152 项回归保护）。
- **M5 配置与 CLI 面**（两个增量，M5b 可裁剪）：M5a = models.json + `cra config`(+test) + 环境变量回退 + 首次引导；M5b = `cra review`（批量/`-o`/`--json`/退出码）+ 全局参数 + REPL 命令族（/help /save /load /undo /context /export /confirm）+ README 同步。

**作业二规划**（Simple Workflow MVP，不提前实现）：代码审查工作流方向——以本项目 Agent 为节点（§2.2 接缝），LangGraph `StateGraph` 编排"编码→审查→修复"或同等流程；顺序执行为 MVP，分支/并行/日志追踪按作业二推荐项递增。

**Backlog（未决功能池，新增功能先入池评审再排期）**：`--log-file` 本地调试日志；`/history` 会话消息摘要；累计 token 用量显示；多智能体协作/反思复核/RAG（作业二本体素材）；可视化编排界面（作业二可选项）。
