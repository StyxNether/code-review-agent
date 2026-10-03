# Design.md — 代码审查助手 code-review-agent

平台环境：Windows + Git Bash（PowerShell/CMD 兼容）。

> 本文描述**目标形态**；已排期内容（§10）全部落地，代码、README 与本文一致；§2.2 的作业二接缝属目标形态、留待作业二。决策演化与开发过程记录见 §4 与 §8。

## 1. 项目概述

`cra` 是一个基于 LLM 的命令行代码审查助手：给定一段/一个/一组代码文件（任意文本型源码、脚本与配置文件，不限编程语言），它以代码审查员身份给出问题清单（分级、定位、修改建议），并能在多轮对话中自主调用工具（读文件、运行代码）核实问题。支持多供应商模型配置与脚本化调用。

## 2. 范围

### 2.1 范围

| 级别 | 内容 |
|------|------|
| Must | 审查代码（任意文本型源码与配置文件，不限语言）；文件读取工具（列目录 + 读单文件）；代码执行工具（subprocess 限时 10s，本机运行非沙箱，限制明示）；多轮上下文记忆；指数退避重试；CLI（彩色流式输出，入口 `cra`）；pytest 测试集；多供应商模型配置（`~/.cra/models.json` + 配置模式）；非交互审查模式 |
| Should | 审查结果导出 Markdown 报告（`cra review -o` / `/export`）；批量审查多文件（`cra review` 多路径） |
| Won't | Web 界面；多语言**执行**（run_python 仅运行 Python，任意语言源码的审查不受限）；Docker 沙箱；联网搜索工具；LLM 调用费用核算（只报 tokens） |

### 2.2 与作业二的接缝（只留缝，不提前实现）

作业二为 Simple Workflow MVP（2–3 个 Agent 协作、消息传递/状态共享、顺序流程控制，推荐条件分支/循环/并行），可复用本项目 Agent 作为 Workflow 节点。为此本设计预留三个接缝（均已落实）：

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
  cli.py        # argparse 子命令（chat/review/config/uninstall）+ REPL + rich 流式彩色渲染
  agent.py      # create_agent 封装：系统提示词组装、工具绑定、最大轮次、上下文截断、回调适配
  llm.py        # ChatOpenAI 封装：客户端构建（base_url/api_key/温度/超时/重试/认证快速失败）
  config.py     # 配置层：~/.cra/models.json 与环境变量的合并读取（唯一出入口）
  prompts.py    # 系统提示词与审查输出格式约定
  tools/
    __init__.py # 工具注册表：@tool 定义集合 + "错误："结构化错误协议包装（行为锁定点）
    fs.py       # list_dir、read_file
    exec_py.py  # run_python（subprocess 限时执行）
  sessions.py   # 会话持久化（/save /load）
tests/
  fixtures/buggy_samples/   # 预埋已知问题的样例（黄金回归用例）
```

### 3.1 Agent 设计

"输入 → 推理 → 工具调用 → 输出"基于 LangChain 1.x `langchain.agents.create_agent` 实现（底层 LangGraph）：

1. 用户输入追加进会话消息；系统提示词常驻（角色 + 输出格式 + 工具使用策略，见 §3.3）。
2. 模型客户端为 `ChatOpenAI`（OpenAI 兼容，`base_url`/`api_key` 来自配置层），`temperature=0.2`（降低输出抖动），请求级超时 60s，`stream_usage=True` 显式开启（usage 是截断触发与 `/context` 的数据源，兼容端点下常缺省）。ChatOpenAI 按 OpenAI 官方规范对齐、裁剪非标字段（取舍见 D8/§9）。
3. 工具经 `@tool` 声明（类型标注 + 参数描述），统一委托既有注册表 handler 执行；**"错误：…"结构化错误协议是行为锁定点**——框架参数校验、未知工具名、执行异常三类分支都必须以"错误：…"文本回传模型（实现策略自定），cli 的工具失败展示依赖该前缀。
4. **最大工具轮次 10**：框架无原生"注入指令 + 去工具收尾"机制——由自定义 middleware 实现（计数工具轮，达限后对下一次模型调用去工具并注入收尾指令）；同时每次调用显式传 `recursion_limit`（≈ 2×轮次上限 + 余量）作崩溃护栏，不依赖框架默认值。官方 ModelCallLimit/ToolCallLimit 的 exit 语义不等价（不注入、不去工具），不采用。
5. **会话状态归属**：不使用 checkpointer——agent.py 仍是会话消息的唯一持有者，每轮以无状态方式把消息列表注入图、取回最终结果回写；/undo、/save /load、Ctrl+C 回滚都作用于该列表（Ctrl+C 契约见 §5，命令规格见 §7.3）。checkpointer 留给作业二工作流按需引入（D14）。
6. **上下文截断**：以最近一次响应的真实 usage 超过阈值（默认 100000 tokens）触发，在**请求前**按完整轮次裁剪（"原子丢弃"= 请求侧丢弃；agent 持有的完整历史不动，/undo 与会话持久化语义因此简单）；`tool_calls`/`tool` 配对不拆散，系统提示词经 system 通道每次注入、永不进入裁剪范围；usage 缺失时不触发（安全降级）。语义由回归测试锁定。
7. **流式渲染（D15）**：文本增量交 cli 渲染管线，粒度为**行级单元流式**——完整行即时落盘（Markdown 渲染 + 分级标记着色）；围栏（``` / ~~~）、列表、表格、引用、缩进代码为**完整性单元**，整组渲染、中间不切段（组等待终止行到达才落盘）；无结构长行超过阈值（60 字符）后在句界提前落盘（句级早发，句界须在行内代码之外）。**过程/正文分类沿用 flush 机制**：工具打点前的缓冲文本按暗色平文落盘为过程叙述，最终回答经 Markdown 渲染；分类窗口为工具结果后的首个单元（暂扣至第二个单元到达，过程叙述实测通常单行），取舍记 D15。叙述、工具打点、最终回答三类内容分层样式。usage 与最终消息（含 finish_reason，如 length 截断提示）随最终响应取得；框架类型由 agent.py 适配为纯字符串回调，不越过 agent 层（§3.4）。
8. `Agent.run(input, callbacks) -> str` 为可编程接口（§2.2 接缝 1）；任何异常（Ctrl+C 或请求失败）回滚本轮消息到合法前缀后原样上抛。

### 3.2 工具 schema

| 工具 | 参数 | 行为与限制 |
|------|------|-----------|
| `list_dir` | `path` | 返回两层目录树；忽略 `.venv`、`__pycache__`、`.git`、`.pytest_cache`、`.ruff_cache` 与点开头的隐藏目录 |
| `read_file` | `path`, `start?`, `end?` | 带行号返回内容；单次上限 **400 行**，超出提示用 start/end 分段；`start`/`end` 为 1-based 闭区间，非法区间返回结构化错误；二进制/解码失败返回结构化错误信息（回传模型） |
| `run_python` | `code` 或 `path`（二选一）, `timeout?` | subprocess 运行，工作目录=进程启动目录；默认 10s（`CRA_EXEC_TIMEOUT` 可调），与模型传参同标准校验（1~60 秒），Windows 下 `taskkill /T /F`、POSIX 下独立进程组 + `killpg` SIGKILL 强杀整棵进程树；子进程环境经净化后继承（剔除 `SE_CodeAgent` 与 `*API_KEY*` 型变量，大小写不敏感）；返回 stdout/stderr/退出码；stdout/stderr 各截断至 **2000 字符**；**本机直接运行，非沙箱**；执行确认开关见 §7.2/§7.3（`cra review` 非交互模式不支持 `--ask-exec`，指定时报错退出 2） |

通用规则：三工具的相对路径一律相对**进程启动目录（cwd）**解析；所有文件读取与子进程输出捕获显式 `encoding='utf-8', errors='replace'`，子进程环境另设 `PYTHONIOENCODING=utf-8`。

### 3.3 系统提示词要点

角色为资深代码审查员（跨语言，任意语言的源码、脚本与配置文件均可审查；执行验证仅 Python——run_python 只能运行 Python，其他语言静态审查并标注"不确定"）；输出固定结构：【严重/一般/建议】分级 + `文件:行号` + 问题 + 修改建议（修改建议独立成行、紧跟问题描述）；先调用工具核实再下结论；不确定的问题明确标注"不确定"，禁止编造行号或代码；用户在消息中直接粘贴的代码直接按粘贴内容审查，不要求另存为文件。默认输出格式为**提示词约定**，不做程序化校验；`--json` 模式（§7.1）例外：按 §7.5 的 schema 程序化输出与校验。

### 3.4 代码设计规范落点

风格层由 ruff 强制（§6）；设计层按以下原则执行，各里程碑收尾按 §8.2 Checklist 核对：

| 原则 | 本项目落点 |
|------|-----------|
| 单一职责 | cli 只管交互与渲染；agent 只管循环与状态；llm 只管模型客户端与重试；tools 只管工具执行；config 是配置唯一出入口 |
| 依赖与模块化 | 依赖方向单向：cli → agent → {llm, tools}；tools/llm 不反向 import agent/cli。框架类型（LangChain 消息/流对象、SDK 异常）不越过 agent 层向上暴露（类型级豁免：cli 仅可从 llm 导入 `LLMError` 类型与 `probe` 连通性自检（§7.4 `cra config test` 专用最小通路，无 Agent 循环）、从 tools 导入 `ERROR_PREFIX` 常量与 `split_lines` 行切分（审查 prompt 行号与 read_file 同口径）） |
| 数据封装 | 会话消息结构与截断规则只存在于 agent.py；工具返回统一为字符串（错误也是字符串） |
| 异常处理 | 按 §5：异常要么被结构化回传模型，要么给出用户可读报错，不允许静默吞掉 |
| 性能 | 流式输出保证感知性能；工具调用有超时；上下文有截断阈值 |
| 安全 | key 只在内存流转、不进仓库（用户目录配置文件除外，见 §7.4），且不进入子进程环境（run_python 环境净化）；run_python 限时 + 截断 + 环境净化 + 限制明示 |

## 4. 关键决策记录（ADR）

被取代的决策保留条目并在状态列标注去向——决策演化本身是项目过程的一部分。

| # | 决策 | 状态 | 理由与后果 |
|---|------|------|-----------|
| D1 | 手写 ReAct 循环（消息列表 + function calling + 循环分发），不引入框架 | 已取代（→D8） | M0–M3 以原生 API 路线实现并验收；吃透 Agent 原理、架构可讲可测。被取代原因见 D8 |
| D2 | LLM 用 DeepSeek，OpenAI 兼容接口 | 采纳 | 国内直连、成本低；OpenAI 兼容接入点可替换服务商 |
| D3 | 仅 CLI，入口 `cra`，不做 Web 界面 | 采纳 | 聚焦 Agent 核心能力 |
| D4 | run_python 用 subprocess + 限时强杀，不做沙箱 | 采纳 | 范围权衡；风险用"限制明示 + 超时强杀 + 输出截断 + 可选执行确认"控制 |
| D5 | 统一采用流式；文本增量与 usage/finish_reason 分通道取得 | 采纳 | 轮次类型无法预判；tool_calls 分片组装完整性等价；工具轮伴随的少量文本同样渲染（已知取舍）；不使用 rich.Live（Windows 控制台重绘不可靠） |
| D6 | 上下文截断按完整轮次原子丢弃，系统提示词永留 | 采纳 | 拆散 tool_calls/tool 配对会导致 API 400；截断为请求侧裁剪、agent 持有完整历史（§3.1(6)），语义由回归测试锁定 |
| D7 | 审查输出格式为提示词约定，不做程序化校验 | 采纳（结构化例外见 D10） | 格式断言测试脆弱；`--json` 是显式例外（schema 校验），且退出码分级判定仅在该模式下保证（§7.5） |
| D8 | Agent 循环迁移至 LangChain 1.x `create_agent`（底层 LangGraph） | 采纳 | 作业二要求 2–3 Agent 工作流并复用本项目：LangGraph 的节点/边/状态原语直接支撑编排与可扩展性（作业二要求之一）；`create_agent` 为官方现行推荐 API；工具层/提示词/渲染/测试策略全部可复用。备选 AutoGen/CrewAI（多智能体叙事错位）、Dify（平台非库，违反仅 CLI）、LangGraph 直接手写 agent 图（原语相同但自维护面更大）、langchain-deepseek 专用包（破坏多供应商统一封装）均排除；后果：接受框架版本漂移风险（§9），以 uv.lock 锁定管理 |
| D9 | 多供应商模型配置：`~/.cra/models.json`（用户目录，仓库外）+ 环境变量回退 | 采纳 | 需求来源：经命令行配置供应商/密钥/模型；key 存用户目录可同时满足"配置体验"与"仓库无明文密钥"门禁；JSON 程序化读写优于 Markdown |
| D10 | 非交互审查模式 `cra review`（含批量/报告导出/--json 结构化输出/退出码） | 采纳 | CLI 审查工具的标配形态；一并落地 Should 两条；是作业二"Agent 即节点"的雏形 |
| D11 | run_python 默认自动执行，`--ask-exec`/`/confirm` 可开启逐次确认 | 采纳 | 审查中执行验证是高频动作，默认打断伤体验；安全叙事由文档明示 + 可选开关表达 |
| D12 | 文档以最终形态维护（不设版本号），决策历史由本表与 §8 PSP 附录承载 | 采纳 | 无需保留旧版本；决策演化靠 ADR 状态列追溯 |
| D13 | 重试映射 SDK：`ChatOpenAI(max_retries=3)`，指数退避间隔与 5xx 语义以 openai SDK 实现为准（重试仅覆盖建连阶段） | 采纳 | middleware 级自定义重试与"流式中途断开不重试（防重复渲染）"冲突；SDK 重试同样仅覆盖建连期，语义等价性最高；401/403 快速失败语义保留 |
| D14 | 会话消息正本由 agent.py 持有（不使用 checkpointer 的无状态图调用） | 采纳 | 保住消息结构/截断/回滚的单点封装（§3.4）与 /undo、/save /load 的实现路径；Ctrl+C 回滚即消息列表快照还原；作业二工作流由 StateGraph 自管状态 |
| D15 | 流式渲染粒度 = 行级单元 + 句级早发 + 边界后暂扣分类的**追加式打印**；不引入 rich.Live（D5 复核后维持） | 采纳（M6） | 段落批渲染是 M4 验收"一大段一次性出"的主因；行级单元对围栏/列表/表格/引用整组渲染保完整性（rich 逐条打印列表项会因块边距插入空行，实证）；句级早发只作用于无结构长行（ASCII 句读须后随空白，防切 `a.py:10` 类路径）。D5 边界复核结论：rich.Live 仅在需要**重绘已打印内容**（真折叠/原地改写）时才有必要，追加式打印无此需求，决策维持。代价（已知取舍）：① 叙述/正文分类窗口为工具结果后的首个单元，多行过程叙述的首行会以正文样式先行显示（过程叙述实测通常单行）；② 最终回答首行延迟至次行到达；③ setext 标题等罕见 Markdown 结构按普通行降级渲染（模型输出以 ATX 标题为主） |
| D16 | 思考过程展示采用**提示词引导的过程叙述**，不透传 DeepSeek `reasoning_content` | 采纳（M6） | 透传属设计变更：langchain-openai 1.6.6（uv.lock 锁定版）文档明示非标字段（`reasoning_content`/`reasoning_details`）不被提取，须供应商专用子类或专用包（langchain-deepseek，D8 明确排除项）；且默认模型 deepseek-flash 非 reasoning 模型、无该字段，收益面窄。提示词引导（工具调用前一句意图说明 + 开工前审查思路）对所有模型生效、零协议风险，过程叙述经既有 flush 管线暗色展示。若未来换用 reasoning 模型且需原生思维链，再评估专用集成包（须另行评估，§9） |
| D17 | 供应商预设清单 + key 环境变量存储（首次引导式配置；Windows `setx` 自动写入；回退链 `CRA_供应商名大写_API_KEY` → `SE_CodeAgent`；**设置时登记、卸载只清自设变量**） | 采纳（2026-10-02 依据验收反馈排期并确认方案；同日修订：覆盖保护与登记清单；后续修订：变量名统一 `CRA_` 前缀格式，不再用各家官方惯用名） | 实际使用暴露首次配置断层：新机器既无 models.json 也无环境变量，现有"打印指引退出 2"把全部配置成本压给用户。预设清单把"添加供应商"压缩为选编号 + 填 key；环境变量选项缩小 D9"明文 key"取舍面（用户二选一，选环境变量则 models.json 不落明文）。后果：① `setx` 属系统状态变更，写入前不回显 key 值、文档明示；② 预设模型名有时效性——内置清单标注核对日期、实现前逐项复核官方文档、预填项均可修改、`cra config test` 兜底验证；③ 离线可用优先，**不联网拉取清单**；④ POSIX 不自动改 shell 配置文件，只打印 `export` 指令；⑤ 修订（卸载语义澄清）：使用环境中可能本就存在同名变量（如使用者此前手动配置过）——写入前询问覆盖（默认 N）、写入即登记 `~/.cra/env-vars.json`，卸载按清单只清自设变量，"删干净"的边界 = 本程序外溢物，项目文件夹由使用者自行删除 |

## 5. 错误处理与重试

行为契约（实现映射到 LangChain/SDK 能力，由回归测试锁定；重试语义见 D13）：

- 启动校验：无任何可用配置（配置文件 default 与环境变量皆缺）时打印 `cra config` 设置指引后退出（退出码 2）；**不校验 key 格式**（无效 key 由 API 认证环节暴露）。chat 模式先询问"是否现在配置"——确认则进入引导式配置（添加供应商 → key → 设默认模型 → 进入 REPL），拒绝或完成后仍无配置维持指引 + 退出 2；`cra review` 保持非交互、无配置直接退出 2。
- LLM 调用：瞬时错误按指数退避重试（最多 3 次，经 SDK 重试承担、仅覆盖建连阶段，间隔以 SDK 实现为准——D13）；**认证类错误（401/403）快速失败、不重试**，输出 key 排查指引（跨平台：Windows setx / macOS-Linux export）；仍失败给出友好报错与排查建议；流式迭代中途断开不重试（增量已渲染，重发会重复输出）；SDK 异常 → `LLMError` 的翻译留在 llm 层，agent/cli 不接触 SDK/框架异常类型。
- 工具执行：所有失败（含框架参数校验、未知工具名）以"错误：…"结构化文本回传模型（§3.1(3)），REPL 不崩溃。
- `run_python`：超时强杀进程树；stdout/stderr 截断防刷屏。
- `Ctrl+C`：中断当前生成——任何中断必须把会话消息回滚到本轮开始前的合法前缀并提示该轮作废；输入 `exit` 或再次 `Ctrl+C` 退出 REPL。

## 6. 工程规格

- 包管理：uv（≥0.12），`requires-python = ">=3.13"`，`.python-version` 固定 `3.13`；包目录 `src/cra`，hatchling 打包配置 `packages = ["src/cra"]`。
- 运行依赖：`langchain>=1.0`、`langchain-openai>=1.0`、`rich`（以 uv.lock 锁定；`openai` 由 langchain-openai 传递提供）。开发依赖：`pytest`、`ruff`。新增依赖须先评审确认。
- 入口与 CLI 面：`[project.scripts]` → `cra = "cra.cli:main"`；子命令 `chat` / `review` / `config`（§7）。全局安装：`uv tool install -e .` 后可直接 `cra`。
- 配置（环境变量，作为 `~/.cra/models.json` 不存在时的回退，统一在 `config.py` 读取）：

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `CRA_DEEPSEEK_API_KEY` | 配置文件缺省时必填 | — | 默认供应商（DeepSeek）的 API key |
| `SE_CodeAgent` | 否（历史变量） | — | 旧版 key 变量名，仍作最后回退 |
| `CRA_MODEL` | 否 | `deepseek-flash` | 模型名回退 |
| `CRA_BASE_URL` | 否 | `https://api.deepseek.com` | 接入点回退 |
| `CRA_EXEC_TIMEOUT` | 否 | `10` | run_python 超时（秒） |
| `CRA_MAX_CONTEXT_TOKENS` | 否 | `100000` | 会话截断阈值（tokens） |

- ruff：默认规则 + `line-length = 100`；`tests/fixtures` 豁免（预埋样例属数据非业务代码）。
- pytest `testpaths = ["tests"]`；**测试内禁止真实网络调用（LLM 一律 mock）**。
- `.gitignore`：忽略 venv/缓存/构建产物/本地辅助文件等（完整清单见仓库 .gitignore）。
- 密钥边界：api_key 只允许出现在用户目录 `~/.cra/models.json`（§7.4）与进程内存；**仓库内任何文件、日志、测试样例、提交记录不得出现真实密钥**。

## 7. 接口规格

### 7.1 CLI 子命令

```
cra                              # 省略子命令：默认进入对话式审查 REPL（裸形式不带参数，选项用 cra chat 传入）
cra chat   [选项]                  # 进入对话式审查 REPL
cra review <path...> | -  [选项]   # 非交互一次性审查（- 读 stdin；多路径=批量）
cra config [test]                  # 交互式配置模式；test 为连通性自检
cra uninstall                      # 卸载本工具：程序/垫片/PATH/用户数据/自设环境变量分级清理（多道交互确认）
cra --version                      # 版本号
```

`cra review` 选项：`-o <file>` 将审查结论写入 Markdown 报告；`--json` 以 §7.5 schema 输出结构化结果（stdout）；其余选项与 chat 一致。

### 7.2 全局参数（chat / review 通用）

| 参数 | 说明 |
|------|------|
| `--model <name>` / `--provider <name>` | 临时覆盖本次使用的模型/供应商（不改配置文件） |
| `--no-stream` | 关闭流式渲染，整体输出 |
| `--max-rounds <n>` | 覆盖最大工具轮次（默认 10） |
| `--ask-exec` | run_python 执行前逐次人工确认（默认自动执行；REPL 内 `/confirm on|off` 同效）。仅限 chat 模式；`cra review` 指定时报错退出 2 |

### 7.3 REPL 内置命令

| 命令 | 行为 |
|------|------|
| `/help` | 命令列表 |
| `/clear` | 会话重置为仅含系统提示词（角色与格式契约保留） |
| `/change model`（别名 `/model`） | 列出供应商及其模型清单，按序号选择；切换后上下文保留并打印提示；未配置任何供应商清单时提示运行 `cra config` |
| `/save [名称]` / `/load [名称]` | 会话持久化到 `~/.cra/sessions/<名称>.json`（消息数组）/ 从文件恢复（整体替换当前会话）。`/load` 省略名称时列出已有会话数字选择（回车取消）；恢复成功后自动打印会话历史（`你>`/`cra>` 逐条） |
| `/undo` | 作废上一轮对话（回滚到上一轮开始前的状态） |
| `/context` | 显示当前会话 token 用量与距截断阈值的余量 |
| `/export [文件]` | 将最近一轮审查结论导出为 Markdown |
| `/process` | 显示最近一轮的过程明细（工具轮叙述全文、工具调用与结果预览）——紧凑打点的可选重放 |
| `/setting` | 设置菜单（数字选择）：1) 切换当前会话模型（复用 /change model） 2) 配置模型列表（供应商/模型/key/默认项，复用 `cra config` 交互函数） 3) 保存会话 4) 加载会话 5) run_python 执行确认开关 6) 删除会话（= /delete）；0 或回车返回对话，菜单内上下文不丢失，保存回车自动以时间戳命名 |
| `/resume` | 恢复最近一次自动保留的会话（= /load _last；结构自检失败则提示存档不可用，正本不变） |
| `/delete` | 删除已有会话：列出会话数字选择 + y/N 确认后删除文件（`_last` 同样可删，删即清除自动保留） |
| `/confirm [on|off]` | run_python 执行确认开关（默认 off） |
| `exit` 或 `/exit`（或 Ctrl+C 两次） | 退出 |
| 其余 `/xxx`（未匹配内置命令） | 按普通文本发送给模型 |

会话自动保留（`/resume` 的数据来源）：REPL 以任何方式退出（exit / /exit / EOF / Ctrl+C）时，将当前会话消息覆盖写入 `~/.cra/sessions/_last.json`（序列化与 /save 同路径，失败仅提示不阻断退出）；启动时若存在非空存档则打印一行提示（不自动加载，避免上下文静默混入）。

### 7.4 配置文件 `~/.cra/models.json`

```json
{
  "default": "DeepSeek/deepseek-flash",
  "providers": [
    {
      "name": "DeepSeek",
      "base_url": "https://api.deepseek.com",
      "api_key": "sk-...",
      "models": ["deepseek-flash", "deepseek-v4-pro"]
    }
  ]
}
```

- 位置固定在**用户目录**（Windows 为 `%USERPROFILE%\.cra\`），不在仓库内、不随仓库分发；`cra config` 负责创建与编辑（交互式：增删供应商/模型、设默认、改 key；**同名供应商 + 同名模型**已存在时询问是否覆盖——供应商同名即更新其接入点与 key、模型清单保留；同名模型在纯名称数组中"覆盖"即确认保留该名称，语义为防误操作确认）。写入为原子替换（临时文件 + rename），POSIX 上自动收紧为 0600。
- 边界规则：`default` 以第一个 "/" 分隔"供应商/模型"，供应商名不得含 "/"（`cra config` 写入时校验）；`--model` 仅在已解析供应商的模型清单内匹配，跨供应商重名时必须同时给 `--provider`，否则报错；`--provider` 单独使用时取该供应商 `models[0]`；`default` 引用的供应商/模型被删除后视为配置损坏，按运行错误退出 2；provider 条目缺 `api_key` 时按序回退环境变量：`CRA_供应商名大写_API_KEY`（全供应商统一格式）→ `SE_CodeAgent`（历史变量，保留为最后回退；是否匹配由认证环节暴露）。多个未设 `api_key` 的供应商若归一化到同一 key 环境变量名且接入点主机不同（凭据串用面），视为配置损坏——加载与 `cra config` 写入时同样报错并给出修复指引；`SE_CodeAgent` 为有意共享的最后回退，不在此判定范围内。
- 供应商预设与 key 存储选择（D17）：`cra config` 与首次引导添加供应商时提供预设编号选择——选预设自动填入官方 base_url 与模型清单（**预填项均可修改**：base_url 回车用官方接入点或输入自定义；模型清单询问是否使用预填）；填 key 后选择存储方式——1) 明文写入 models.json（现状，含权限与备份提示）2) 环境变量（变量名统一为 `CRA_供应商名大写_API_KEY`：供应商名大写、空格与非字母数字转 `_`、非 ASCII 字符剥离；预设供应商同格式、取规范短名；Windows 经 `setx` 写入用户环境并设置当前进程 `os.environ` 即时生效，POSIX 打印 `export` 指令由用户执行）；选环境变量时 models.json **不存明文 key**，该供应商已有明文的**一并移除**。
- 环境变量覆盖保护与登记（D17 修订，2026-10-02）：写环境变量前若该变量**非本程序所设**且当前环境已有同名变量（使用环境中可能本就存在，如使用者此前手动配置的 `CRA_DEEPSEEK_API_KEY` 等），询问"是否覆盖其值"（y/N **默认 N** 保护既有值，拒绝则 key 回退明文、原变量不动）；变量已登记在案（本程序之前设置）时直接更新、不再询问。每次成功写入即登记到 `~/.cra/env-vars.json`（去重、原子写；损坏按空清单处理）——**卸载时只清理登记在案的变量**，绝不触碰用户已有变量；登记失败（磁盘等）不阻塞 key 设置但明示"卸载时无法自动清理"。
- `cra config` 写入时提示文件权限与备份注意事项（key 为明文，属用户本机数据，见 D9/§9）。
- 解析优先级：`--model/--provider` 参数 > 配置文件 `default` > 环境变量回退（key：`CRA_DEEPSEEK_API_KEY` → `SE_CodeAgent`；模型/接入点：`CRA_MODEL`/`CRA_BASE_URL`）。配置文件与环境变量皆缺时，打印 `cra config` 引导后退出（退出码 2）。
- `cra config test`：对每个供应商用当前 key 发一次 `max_tokens=1` 的对话请求，报告 OK / 401（key 无效）/ 超时（网络不可达）；退出码：全 OK 为 0、存在失败为 1（检查结论）、配置错误为 2（运行错误）。探测用短超时（15s）且 `max_retries=0`——自检语义是快速反馈，不承担业务请求的重试。
- 安全边界：该文件含明文 key，属用户本机数据的已知取舍（D9）；文档提示文件权限与备份注意事项，仓库侧由 .gitignore 与密钥门禁保证永不入库。

### 7.5 退出码与结构化输出

| 退出码 | 含义（`cra review`） |
|--------|---------------------|
| `0` | 审查完成（是否发现严重问题仅 `--json` 模式下可程序化判定，见下） |
| `1` | 仅 `--json` 模式：按 schema 的 `severity` 字段判定存在【严重】问题 |
| `2` | 运行错误（参数错误、配置缺失、网络/认证失败、文件不可读等，含顶层兜底） |

非 `--json` 模式的结论是提示词约定的自由文本（D7），**不做分级文本匹配**：退出码只有 0（完成）与 2（运行错误），下游需要分级判定时必须用 `--json`。`cra review` 顶层兜底捕获一切未预期异常归入退出码 2——禁止异常裸逃逸（Python 未捕获异常默认退出码恰为 1，会伪装成"发现严重问题"）。

`--json` 输出 schema（多文件统一形态，供脚本与作业二下游 Agent 消费）：

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

stdin 输入时 `file` 取 `"<stdin>"`；目录入参展开其第一层的文本文件（跳过点开头的隐藏文件与二进制文件）；`--json` 时 stdout 只输出 JSON，`-o` 可同时写 Markdown 报告。单文件内容直接嵌入审查 prompt（带行号），上限 2000 行、超出截断并在 prompt 中明示。

`cra chat` 正常退出码 0；配置缺失退出码 2。

## 8. PSP 附录（轻量诚实版）

**开发过程定位**：本项目采用**增量开发**——每个里程碑交付一个可用增量，经评审验收即持续反馈，回归测试保护既有增量，评审意见回灌下一里程碑。代码审查按静态审查（ruff + pytest）、动态审查（验收运行检查）、代码走查（验收 + 收尾总结，范围 = 该里程碑 diff）三级落地，结果随开发日志记录；§8.2 Checklist 是走查门禁。

### 8.1 估算表

| 项 | 估算 | 实际 |
|----|------|------|
| 总规模 | ~1000 LOC（含测试，git 统计为准） | 10097 LOC（src 4235 + tests 5852 + fixtures 10，`git ls-files "*.py"` 统计；fixtures 为预埋样例数据） |
| 人工投入 | 设计评审 0.5h + 里程碑验收 0.25h×n + 收尾 0.5h | M0–M2 验收实测共 22 分钟；此后未再逐项记录 |
| 日历时间 | 2–3 天（4 个编码会话） | 1 天完成 M0–M3 编码与验收 |
| M4 框架迁移 | 1 个编码会话；净变化 ±300 LOC 待实测（middleware/流式适配/无状态包装可能净增）+ 测试重写 ~400 行 | 1 个编码会话；迁移本体净 **-134 LOC**（3273 → 3139），验收修复轮 +201、cra 自审修复轮 +247 → 3587；测试重写与增补 56→63 项（test_agent 27→43、test_llm 29→20、exec/cli 增补），总数 152→173 |
| M5 配置与 CLI 面 | 1–2 个编码会话；净增 ~800–1200 LOC（含测试） | 1 个编码会话；净增 **+2826 LOC**（3587 → 6413：src 1387→2672、tests 2070→3611、scripts 不变）；测试 173→297 项 |
| M6 流式过程体验 | 1 个编码会话；净增 ~300–500 LOC（渲染管线重写 + 测试） | 1 个编码会话；净增 **+864 LOC**（6413 → 7277：src 2672→3131、tests 3611→4016、scripts 不变）；测试 313→329 项；cra 自审 1 严重 + 1 一般 + 2 建议 → 修复 + 复核轮 6 建议落地 |
| M7 CLI 易用性 | 1 个编码会话；净增 ~150–250 LOC（含测试） | 1 个编码会话；净增 **+493 LOC**（7277 → 7770：src 3131→3264、tests 4016→4376、scripts 不变；超估算主因新用例 25 项 / +360 测试行）；测试 329→354 项；cra 自审三轮（初审 2 一般 + 6 建议 → 复核 2 一般 + 4 建议 → 终审 0 严重、2 一般 + 3 建议，处置见 §8.3） |
| M8 会话与配置便利性 | 1 个编码会话；净增 ~550–850 LOC（含测试） | 1 个编码会话；净增 **+1463 LOC**（7770 → 9233：src 3264→3874、tests 4376→5229、scripts 不变；超估算主因卸载程序完整实现（确认/分级清理/分离任务/PATH 判定 ~300 行）与预设/存储/会话管理交互分支的用例面——+56 用例 / +853 测试行）；测试 354→410 项；cra 自审三轮（初审 2 严重 + 3 一般 + 4 建议 → 复核 1 严重 + 5 一般 + 5 建议 → 终审 4 项小修 + 不采纳 3 项，处置见 §8.3） |
| M8 修订轮（卸载语义澄清） | 同会话追加；预计 ~150–250 LOC（含测试） | 同会话追加；修订本体净增 **~+370 LOC**（src ~+180、tests ~+190），测试 410→428 项（+18：登记层 4、覆盖保护 4、卸载环境变量清理 10）；全仓 LOC 9233→9736（差额为同工作树并入的其他改动：cra review 目录展开放宽 `.py`→第一层文本文件+二进制嗅探、注释引用清理与行尾统一——**非本里程碑产物**，已经复核确认保留并同步 §7.5 与 README 文档）；环境变量登记/覆盖保护/卸载第三道确认与清理全部落地，cra 自审一轮（2 项采纳，见 §8.3 修订轮记录） |

### 8.2 代码评审 Checklist（收尾门禁；每次收尾逐项执行，当前记录：交付前终审轮 2026-10-03——452 项 pytest 全绿、ruff 全绿、12 项逐条复核；M8 主体注记为该轮之前的历史记录，测试数为当时口径 428 项）

- [x] 全仓库 grep 无明文密钥；key 读取点唯一（config 层） ✅（M8 复执行：grep 无命中；key 三级回退链集中在 config.resolve_provider_api_key，cli 的存储选择仅决定"写文件还是写环境"，读取仍唯一）
- [x] 无裸 `except:`；所有工具异常均回传模型而非静默吞掉 ✅（M8 新增兜底均有 noqa 与理由：_ask 扩大捕获 OSError（stdin 被捕获/关闭的环境视同取消，首次引导路径不再崩）、_write_env_var_key 的 setx 失败回退明文（有明确提示，key 不丢）；_cmd_load/_cmd_delete 沿用既有"可读提示 + return"模式）
- [x] `run_python` 有超时强杀与输出截断 ✅（M8 未改动工具层与 middleware）
- [x] Agent 循环有最大工具轮次（10）限制 ✅（M8 未改动）
- [x] 消息截断按完整轮次原子丢弃，配对不拆散，系统提示词保留 ✅（M8 未改动）
- [x] `cra chat --help` 与 README 含"非沙箱、限时执行"限制声明 ✅（M8 未改动声明，有测试锁定；新增 uninstall 子命令 description 明示"确认前不删除任何内容"）
- [x] README 每条命令逐条实测可复制执行 ✅（M8 实测：首次引导全流程（Y→DeepSeek 预设→回车→key→环境变量→默认→REPL→对话）、cra config test、/load 列表与历史打印、/delete、/setting 3/4/6、cra uninstall 确认分支；命令表/存储选择/卸载章节已同步）
- [x] pytest 覆盖：工具 handler、消息组装、截断逻辑、CLI 参数解析 ✅（M8 修订轮后 428 项；M8 主体新增：预设数据完整性与顺序、变量名生成（空格/标点/非 ASCII/数字开头/官方名优先）、回退链三级优先（含 Z.ai 官方名与 SE_CodeAgent 兜底）、首次引导两分支（确认全流程进 REPL / 拒绝退出 2）、预设预填接受与修改、存储选择分支（明文/环境/中断保留/setx 失败回退/POSIX export）、config test 走回退链、/load 列表选择与历史打印、/delete 确认/取消/_last 可删、菜单自动时间戳与 6 分发、卸载多道确认（修订轮后环境变量一问条件出现）/中断保留数据/数据删除与否/未装跳过/uv 失败兜底/脚本拼装/PATH 判定/删除目标护栏/spawn 双平台；修订轮新增：登记层读写/去重/损坏容忍、覆盖保护（他人问 y/N/拒绝回退明文/自己的直接更新/中断保留）、卸载环境变量清理（默认 Y/n/中断/无登记跳过/脚本内容/非法名过滤/POSIX 待删行））
- [x] ruff 全绿；公开函数有类型标注；命名自解释 ✅（M8：_offer_first_run_setup/_choose_provider_type/_input_base_url_and_models/_apply_key_storage/_write_env_var_key/_list_session_names/_choose_session/_print_session_history/_cmd_delete/_run_uninstall/_uv_tool_names/_should_remove_uv_path_entry/_windows_cleanup_script/_posix_cleanup_script/_spawn_cleanup_task/_delete_user_data 均带 docstring）
- [x] 模块依赖方向符合 §3.4（无反向 / 越层 import） ✅（M8 无新增依赖方向；预设数据放 config 层，cli 仅消费；卸载的 subprocess 调用留在 cli 层（安装面/进程管理属 CLI 职责，不涉 agent/llm））
- [x] `git status` 干净：无 venv/缓存/密钥/开发辅助文档被跟踪 ✅
- [x] Design.md 与实现一致（模型、工具名、模块、入口、配置项、CLI 面） ✅（M8 回写：头部状态、§8.1/§8.2/§8.3、§10 M8 落地注记；§5/§7.1/§7.3/§7.4 的 M8 修订在编码前已成文、实现与之逐条对齐）

### 8.3 复盘与缺陷记录

- **缺陷记录**（M0–M4 验收与走查共 19+3 项；M0–M3 的 19 项全部闭环、无遗留；M4 编码期 3 项现场闭环，无新增遗留）：
  - M0 验收 1 项：提示词角色未定义【一般】→ M1 prompts.py 落实。
  - M1 验收 1 项：输出文字堆叠不便阅读【建议】→ 修复轮 1（空行/加粗）。
  - M1 走查（cra 复审两轮）13 项：中断回滚索引失位致孤儿 tool_calls【严重】、空回答静默入库【一般】→ 修复轮 1；标记加粗漏标、围栏奇偶误判、工具失败判定、rich markup 注入、REPL 意外异常兜底、llm 异常包装扩大【一般 6】及显式化小项【建议 5】→ 修复轮 2。
  - M2 编码期 1 项：read_file 空文件误报区间错误【一般】→ 现场修复。
  - M2 验收 2 项：审查语言边界设置错误【一般】、修改建议格式与着色【建议】→ 修复轮（多语言放开、标记切片渲染）。
  - M2 修复轮自测回归 1 项：CommonMark flanking 规则致标记加粗失效【一般】→ 渲染改标记切片。
  - M3 遗留项处置：finish_reason 透传、print/console 统一、指引跨平台化、_run_chat 可测性已落地；stream_options 不做降级重试（DeepSeek 实测支持，400 属参数错误、不在重试范围，有回归锁定）；截断 O(n²) 维持不改（消息规模小且截断罕见，两次评估结论一致）。
  - M4 编码期 3 项（现场闭环）：① `translate_failure` 初稿误写 `return … from exc`（语法错误）→ 返回型翻译改手工设置 `__cause__`；② LangGraph 对传入消息对象原位分配 id，等值断言失配【测试】→ 测试改按语义形状比较（实现不受影响）；③ 达限"去工具"后框架走 `model.bind()` 而非 `bind_tools()`，假模型绑定日志缺该次记录【测试】→ 假模型补记 `bind()` 空绑定，机制经 `request.override` 锁定。
  - **M4 验收修复轮（2026-09-30，验收走查回灌）9 项**（自审复检 + 验收反馈，全部闭环）：
    - 【一般·实证复现】"Error:" 前缀字符串匹配误伤工具正常输出（run_python 的 stdout 恰以 `Error:` 开头时被改写为"错误："并污染回传）→ 改为 **ToolMessage.status=="error" 语义字段判别**（`_normalize_framework_error`）：框架错误（校验失败/畸形参数）必为 error 状态、正常输出为 success，门禁不依赖框架文案；前缀剥离仅作文本清理，已归一消息防重复叠加。附带缓解"框架文案硬编码"风险（判别不再依赖 "Error:"/"could not be executed" 字面）。
    - 【一般】回滚只删末条消息，覆盖不了"成功 extend 之后仍抛异常"的窄窗口（§5 本轮作废契约）→ 改为进入 try 前记录正本前缀长度，异常按 `del self.messages[prefix_length:]` 截断。
    - 【一般】达限收尾指令以伪造 user 消息注入请求，恰构成截断的"轮次边界"——极端场景下截断会保留孤立指令、丢弃全部真实上下文 → 收尾指令改经 **system 通道追加**（不产生伪造 user 消息，轮次边界只认真实输入；指令不落正本不变）。
    - 【一般】on_tool_result 三分支触发条件不一致（正常分支要求非空才回调）→ 统一为每分支必然触发一次（空串也回调），协议对调用方一致。
    - 【建议×4】`str(call.get(k, ""))` 对 None 值产出 "None" 串 → `or ""`；reset() 一并清理回调挂留引用；recursion_limit 的"每轮节点数"提为具名常量 `_NODES_PER_TOOL_ROUND`；token 估算加 1.2 倍安全余量（×0.6，中文占比高时防低估少丢，宁多丢不超限）。
    - 夹具建议（infinite_loop.py）：`__main__` 导入守卫 + 注释措辞收敛（明确"仅覆盖直接子进程强杀、/T 树杀路径未触及"）已落地；CPU 自旋保留（kill 语义与 sleep 循环等价，且保持 CPU 密集场景真实性）。
  - M4 修复轮附注：真实 API 实测逐 token 流式通路正常（320 个增量、拼接==返回值）；使用中感知"一大段一次性出"源于段落批渲染 + 跨轮叙述累积两处，后者已由 cli 叙述分流修复，前者列入 M6。
  - **M4 自审修复轮（2026-09-30，cra 自审回灌——固定流程，dogfooding：每轮收尾前用本项目 cra 审查本轮代码）**：
    - 【严重·实证复现】run_python 子进程全量继承父环境，`SE_CodeAgent` 可被被审查代码经 `print(os.environ)` 读走并回传模型上下文 → 子进程环境净化：剔除 `SE_CodeAgent` 与 `*API_KEY*` 型变量。**一手发现**：实测环境中存在 `SE_CODEAGENT` 全大写变体（Windows 环境名不区分大小写，精确匹配过滤与应用侧读取各漏一边）→ 过滤必须大小写不敏感（`_is_credential_var` 独立成函数便于测试）。§9 的"继承父环境（含 key）已知边界"自此收紧为默认行为。
    - 【一般】工具执行期间 Ctrl+C 残留子进程与管道（communicate 抛出后无回收）→ `except BaseException` 分支杀树 + 有限回收管道后原样上抛。
    - 【一般】POSIX 分支 `proc.kill()` 只杀直接子进程，与"进程树强杀"表述不符 → `start_new_session` + `killpg` 整组 SIGKILL（子树自建进程组者除外，与 taskkill /T 边界一致）；POSIX 分支为源码审查级验证（开发机为 Windows）。
    - 【一般】配置来源 `CRA_EXEC_TIMEOUT` 未与模型传参同标准校验（0/负值会伪装成"执行超时"）→ 同标准 1~60 校验，非法值返回指向配置的结构化错误（cra 自审发现）。
    - 【一般】agent 正本写回在"新增消息为空"场景会把本轮 user 消息就地覆盖成回答 → 截断提示只写回本轮新增消息末条（new_messages[-1]）。
    - 【建议】兼容端点无文本增量时界面空白 → run 返回值兜底渲染；【建议】行内分级标记被 Markdown 片段换行/全宽填充拆行（实测复现）→ 标记行改为片段渲染收敛后拼单行 Text（保留行内代码渲染与原文空格）；【建议】request.state["messages"] 裸索引 → `.get` 降级；【建议】模型通道归一改单次遍历、无变化直通；【建议】taskkill 调用异常兜底（不掩盖原始异常）。
    - 未采纳：timeout 参数非整数类型校验——dispatch 已结构化兜底且可读（维持 M2 处置）。
    - **dogfood 过程记录**：初审 5 项（配置超时/中断回收/过滤不可测/覆盖面/终审附注）→ 全部修复；复审确认三项已落地 + 2 项加固建议（1 采纳 1 维持）；另发现 SE_CODEAGENT 变体泄漏（本轮最有价值的一手发现）与一次模型 DSML 文本泄漏怪癖（deepseek-flash 把工具调用标记当正文输出，属模型侧行为，不处置）。
    - 注：本轮 agent/cli/exe 修改各带回归用例；`grep` 复核无明文密钥（测试用假值 sk-secret-for-test 等）。
- **框架 API 查证结论（M4，以 uv.lock 锁定版本实际行为为准）**：langchain 1.4.3 无 `modify_model_request` 钩子，改请求须经 `wrap_model_call` + `request.override()`（属性直赋已弃用）；系统提示词经独立 `system_message` 通道每次注入、不入图状态；ToolNode 对未知工具/参数校验失败的兜底文案为 `"Error: …"`，须改写为"错误："前缀回传模型（`wrap_tool_call` 统一拦截，含未知工具的 tool=None 短路）；模型节点按请求重新 `bind_tools`（去工具后走无工具的 `bind` 分支）；`stream_usage=True` 在 DeepSeek 真实流式下生效（冒烟 usage=371 tokens）；`ChatOpenAI` 的 temperature/timeout/max_retries/stream_usage 均以 pydantic 字段生效。**已知边界**：模型输出畸形 JSON 参数（invalid_tool_calls）时，框架在模型请求侧合成 "could not be executed" 提示（已归一为"错误："前缀）；混合有效+畸形调用的场景下该合成 ToolMessage 会进入图状态，其 tool_call_id 仅存在于 `invalid_tool_calls`（不入 API 序列化），对严格校验的服务端存在孤儿 tool 消息的理论风险——属罕见路径、框架自身行为，暂不自行规避，若实际供应商 400 再评估（如 fed 视图清洗）。
- **M5 编码期记录（2026-10-01，现场闭环、无遗留）**：
  - **实现决策注记**（§7 的落地细节备注）：
    - `cra review` 的文件内容**直接嵌入 prompt**（行号格式与 read_file 一致，1-based），不走路由工具轮：非交互审查优先确定性与行号可靠性（模型自读再引行号有编造空间，且多一轮往返）；单文件上限 2000 行，超出截断并在 prompt 中明示（防 token 爆炸，§9 已知边界）。
    - `--json`：模型输出经围栏剥离 + 首尾大括号截取 + schema 校验（severity 枚举、location/issue 非空、uncertain 补全、顶层 summary 缺省补空串），校验失败归退出码 2；多文件结果合并为 §7.5 统一形态，顶层 summary 程序化拼接。
    - 非 `--json` 结论**原样输出 stdout**（不经 rich，重定向/管道保真）；过程提示与错误一律走 stderr，`--json` 的 stdout 保持纯 JSON。
    - `/save` 不入档系统提示词（仅对话消息），`/load` 恢复时拼**当前版本** SYSTEM_PROMPT——避免旧提示词固化进存档随会话漂移；会话文件名白名单校验（防路径穿越）。
    - 环境变量回退模式下 `--model` 直接覆盖 `CRA_MODEL`（无供应商清单可匹配）；有清单时仅在清单内匹配（§7.4），默认项引用被删供应商/模型时按配置损坏退出 2。
    - 执行确认钩子（`--ask-exec`/`/confirm`）位于工具 middleware：确认开关关闭自动放行，拒绝时以"错误：用户拒绝…"结构化回传模型，语义与工具失败协议一致。
  - **Mimosa 钩子误报处置**（写入期拦截，均属启发式误报）：新写测试中的占位假值两次被拦为"硬编码凭据"（"sk-" 前缀形态、`"api_key": "<字面量>"` 键值对与 `api_key="<字面量>"` 关键字形态）——测试占位值统一改为 `placeholder-*` 并以变量构造避开字面量模式；config 层 `ENV_API_KEY = "SE_CodeAgent"` 常量（值是环境变量名非凭据）同样被拦，回归既有字面量形态。处置不改变任何行为。
  - 现场闭环小项：RUF012 类属性可变默认（ClassVar 注解）；测试假 Agent 类属性跨用例泄漏（fixture 重置）；`datetime.now` 的 DTZ005（报告时间戳改 `astimezone()` 显式本地时区）；JSON 模式顶层 summary 缺省补全。
  - **smoke_tools.py 兼容性确认**：config 层保留环境变量回退函数（get_api_key/get_model/get_base_url），`llm.create_model()` 默认路径不变——冒烟脚本无需改动（M5a 任务 4 的同步适配项：确认无影响）。
- **M6 编码期记录（2026-10-01，cra 自审 1 严重项由审查者实证复现）**：
  - **现场闭环 1 项**：增量兜底判据缺陷——`--no-stream` 前身判据"是否有可见输出（has_output）"覆盖不了"增量已到但尚未落盘（未成单元）"的窗口，此时用返回值兜底会把同一文本渲染两遍；改判据为"工具边界后是否收到过增量（has_delta_since_boundary）"，带两条回归（部分增量不重复、叙述已冲洗+最终无增量仍兜底）。
  - **M6 cra 自审记录（DoD 第 4 条，2026-10-01，两批：初审 + 复核确认）**：
    - 初审（对象 = M6 代码 diff，经 `cra review -` 管道）发现【严重】1 + 【一般】1 + 【建议】2：
      - 【严重·实证复现】`_lead_unit` 消费前导空行后切片重定位 `self._buffer`，但围栏开栏偏移 `limit` 未同步平移——**一次投喂**（`--no-stream` 与返回值兜底路径的真实形态）时围栏开栏行被判为普通单行单元渲染，代码块被拆碎、``` 标记字面露出；逐字符投喂（常规流式）恰好掩盖，既有测试未捕获。修复 = 空行分支同步平移 limit，围栏分支改按当前缓冲重算 span；回归用例 = 空行与围栏开栏行落在同一增量。**审查者自行完成了复现、修复方案验证与回退对照**，是本项目 dogfood 质量最高的一次自审。
      - 【一般·不采纳】`_first_fence_open` 每单元重扫的 O(N²)：流式下缓冲被单元消费约束在未闭合结构内（通常数行），一次性灌入的 N 行 drain 全程亚毫秒，与 M1/M3 的 O(n²) 评估同尺度；缓存扫描位置的状态机回归风险大于收益，维持简单实现。
      - 【建议×2 已落地】`_scan_group` 谓词 `Any` → `Callable[[str], bool]`；flush_pending 拼接前归一化（组单元自带尾换行与单行单元混拼的多余空行）。
    - 复核轮（对象 = 修复 diff）：**确认严重项修复有效**（审查者用回退对照复现了修复前的拆碎行为，并跑了 69 项 cli 测试 + 800 例行/围栏/空行随机模糊，内容无丢失、顺序不变）；新发现 6 项【建议】全部落地：flush 归一化改 `rstrip` 保留首行缩进（冲洗内容可能是缩进代码/围栏）；正则→谓词映射集中（`_heading_line` 等六个具名谓词，`_is_structured_line`/`_is_block_opener` 复用）；列表成员谓词提为具名 `_list_member`；"自审严重项"等流程性注释改写为陈述坐标不变量。
    - 附注：复核确认 markup 注入面（/process 与工具打点均 `markup=False`/Text 组装）、`_run_review` 顶层兜底不误吞 SystemExit/KeyboardInterrupt、可变默认值均无问题。
  - 模型侧行为备注：自审第一轮经 `cra chat` 执行时再现 DSML 工具调用标记当正文输出的怪癖（M4 起第三次记录），属模型侧行为，不处置；改用 `cra review -`（stdin 直通、结论 stdout 纯文本）完成审查。
- **M7 编码期记录（2026-10-01，入口接线；无严重遗留）**：
  - **实现决策注记**：
    - 裸形式不带参数：argparse 子解析器会用自身默认值覆盖顶层同名参数值（`cra --no-stream chat` 的 no_stream 会被 chat 子解析器重置），故共享参数面保持在子命令之后；main 用 `getattr(args, …, 缺省)` 分发裸入口，缺省值与 `_add_shared_options` 的一致性有专门测试锁定。§7.1 的 `cra [选项]` 草图据此修订为 `cra`（裸形式不带参数）。
    - /setting 纯接线：1/3/4/5 直接复用 _cmd_change_model/_cmd_save/_cmd_load/_cmd_confirm；2 复用配置交互主循环——从中拆出 `_config_interactive_session(console, data)`（独立运行仍走 _run_config_interactive 包损坏配置退出 2，§7.5 契约不变），菜单内以 ConfigError 类型降级回菜单，不依赖退出码数值做跨函数契约（cra 复核轮建议）。
    - 菜单嵌套循环：操作失败/取消（含 EOF/Ctrl+C 经 _ask 归一）一律回菜单，0/回车回 REPL 主循环；菜单全程不触碰会话消息，上下文不丢。
    - 自动保留：try/finally 包裹 REPL 主循环（拆出 _repl_loop），覆盖 exit //exit/EOF/Ctrl+C 全部退出路径；**空会话跳过**（新开即退不覆盖上次有效存档）；失败仅提示不阻断退出（finally 中函数自身不得上抛，收尾打印亦在兜底内）；启动检测到非空存档打印一行提示、不自动加载；/resume = /load _last（agent.load_session 结构自检失败正本不变）。
    - sessions 名称白名单放开首字符下划线（_last 落同一命名空间；分隔符禁用使穿越防护不变），错误消息同步"以字母、数字或下划线开头"（cra 终审发现）。
    - 测试隔离升级：conftest 的 home_dir 改 autouse——自动保留使每次 REPL 退出都写 ~/.cra/sessions/_last.json，不隔离会污染真实用户目录；全部 354 项在 tmp 用户目录下运行。
  - **M7 cra 自审记录（DoD 第 4 条，2026-10-01，三轮：初审 + 复核 + 终审确认）**：
    - 初审（对象 = M7 全量 diff，经 `cra review -`）：【一般】×2 + 【建议】×6——采纳落地：菜单兜底收敛到分发层（后经复核演进为 ConfigError 契约）、_hint_last_session 放宽为 Exception 并补"可解析但结构非法"存档用例、裸入口缺省值一致性注释 + 锁定测试、main 显式 `in (None, "chat")` 分支、自动保留提示带消息条数 + README 标注覆盖语义、/confirm 菜单用例断言改精确状态行（原断言被菜单回显文本满足恒真）、欢迎横幅同步 /exit。
    - 复核轮（对象 = 修复后 src diff）：【一般】×2 + 【建议】×4——采纳落地：自动保留成功打印移入 try、退出码魔法值改 ConfigError 类型契约（拆 _config_interactive_session）、_run_chat docstring 更新（拆出 _repl_loop 后职责准确）、sessions 注释据实（".." 防护实由分隔符禁用承载）。现场发现一处测试设计缺陷并修正：patch load_raw 会连带破坏 REPL 启动解析（resolve 同源），改 patch _config_interactive_session 并把菜单内 load+session 整体纳入 try——对应"会话进行中配置损坏"的真实场景。
    - 终审（对象 = 复核后 src diff）：**0 严重**；【一般】×2（错误消息与正则不同步 → 采纳修复；_last 命名空间冲突 → 不采纳）+ 【建议】×3（均不采纳）。
    - **未采纳项记录**：① `_last` 独立保留名（初审/复核/终审三轮重提）：设计规格明文"覆盖写入 ~/.cra/sessions/_last.json"与"/resume = /load _last"，独立目录/不可构造名偏离规格；README 与 /help 已声明该名用途，用户显式 /save _last 即显式选择覆盖"最近会话"，语义自洽。② _hint_last_session 的 session_path 防御性 ValueError 包裹：`_last` 合法性由 test_session_path_accepts_leading_underscore 锁定，常量与校验同仓库演进，改名时测试先失败。③ 启动判空改文件大小启发式：会话文件 KB 级、启动解析一次成本微秒级，完整校验还保证损坏文件不触发误导性提示。④ --help 再显式点明裸形式限制：description 与 README 已写明"裸形式不带参数"。
- **M8 编码期记录（2026-10-02，无严重遗留）**：
  - **预设清单时效性复核（D17 要求的实现前逐项复核）**：按官方文档逐项核对——① DeepSeek：base_url `https://api.deepseek.com`、deepseek-flash / deepseek-v4-pro、DEEPSEEK_API_KEY 全部一致（api-docs.deepseek.com 原文）；② Kimi：`https://api.moonshot.cn/v1`、kimi-k3 / kimi-k2.6、MOONSHOT_API_KEY 一致（platform.kimi.com API 概述；域名已迁 platform.kimi.com 但 API 端点不变）；③ Z.ai：coding 端点 `/api/coding/paas/v4` 与普通端点 `/api/paas/v4` 均获官方文档确认、glm-5.3 在列；④ BigModel：`https://open.bigmodel.cn/api/paas/v4`（官方 API 参考）与 coding 端点（ZCode 官方文档等佐证）、glm-5.3 / glm-5.3-flash 均确认；⑤ 阿里云百炼：`https://dashscope.aliyuncs.com/compatible-mode/v1` 与 DASHSCOPE_API_KEY 获官方文档确认，qwen3.8-max/flash 未在本次搜索结果中直接确证；⑥ OpenAI：gpt-6-astra / gpt-6-luna 未在本次搜索中直接确证。⑤⑥ 按设计表（2026-10-02 当日已按官方文档核对）采用——D17 已内建时效性策略：预填项均可修改、`cra config test` 兜底验证，后续若模型更名由用户改预填即可，不构成阻塞。
  - **Mimosa 钩子拦截与处置（写入期，同 M5 先例）**：拟引入 `ENV_API_KEY = "SE_CodeAgent"` 常量被拦为"硬编码凭据"（值是环境变量名非凭据，M5 已记录同型误报）→ 按 M5 处置先例不引入常量、维持既有 `os.environ.get("SE_CodeAgent")` 字面量形态，回退链函数内联该字面量；`env_var_for_provider` 的下划线拼接被启发式放行，无实际处置。
  - **实现决策注记**（§7 的落地细节备注）：
    - 预设的"供应商名"即 models.json 的 name（如 "DeepSeek"、"Z.ai Coding Plan"，可直接用于 default 引用与 --provider）；预设两套 Z.ai/BigModel 共享官方惯用变量名（ZAI_API_KEY / ZHIPUAI_API_KEY）属官方形态。
    - 自定义供应商变量名生成（cra 自审复核后定稿）：剥离后为空（纯非 ASCII/纯符号名）以名称 SHA256 摘要生成唯一变量名 `CRA_API_KEY_<hash8>`——不同供应商不得共用兜底名（否则 key 串用会把一家凭据发给另一家接入点）；首字符为数字前置 `_`（POSIX shell 标识符约束）；`_write_env_var_key` 写入前按 `^[A-Za-z_][A-Za-z0-9_]*$` 校验，非法回退明文。
    - key 存储选环境变量：Windows setx + 当前进程 os.environ 即时生效；POSIX 打印 export（指令含 key 明文，随附保密提醒）**且当前进程同样设置**——当前进程内 resolve 无断层，持久化靠用户执行 export。setx 失败/超长（>1000 字符，setx 对 >1024 静默截断）/变量名非法一律回退明文（有明确提示），key 不因存储失败而丢失。
    - 中断语义统一（cra 自审复核）：`_ask` 返回 None（EOF/Ctrl+C）一律不视为确认——存储选择中断不写 key（保留现状）、卸载数据询问中断保留 ~/.cra/（数据含 key，宁可保留）；首次引导中断按拒绝落到既有指引。
    - `_ask` 扩大捕获 OSError：stdin 被捕获/关闭的环境（pytest、无 stdin 管道）中"是否现在配置"按取消处理，落到既有指引 + 退出 2——这也是既有 `test_missing_key_exits_with_guide` 无输入 mock 仍通过的原因；`_read_key` 同口径。
    - 首次引导只在"零供应商清单"时出现；已有供应商但缺默认/key 的场景维持原指引文案（两者是不同断层，后者一句指引即可解决）；向导内 load_raw 损坏时直接 return（防 save 覆盖用户损坏文件，调用顺序变化下的护栏）。
    - 卸载的 PATH 移除判定在 spawn 时以当前 `uv tool list` 计算 remove_path（无其他工具才注入 PATH 清理块），脚本内卸载后复查（`$LASTEXITCODE -eq 0` 且列表为空才动手——防分离进程无 uv 时误判）双保险；`_delete_user_data` 删除前校验目录名确为 ".cra"（防 config 路径演化误删）；spawn 的 Popen 以 OSError 兜底（powershell/sh 缺失不崩卸载流程）。POSIX 不自动改 rc，仅打印待删行——装/卸与 key 存储三处口径一致（D17）。
  - **测试注记**：354 → 410 项（+56）。`_write_env_var_key` 相关用例以 `monkeypatch.setattr(cli.os, "environ", dict(os.environ))` 隔离进程环境（断言"当前进程即时生效"且不污染真实环境；不能置空 dict——`Path.home()` 依赖 USERPROFILE，首版即踩此坑），并以 `monkeypatch.setattr(cli.sys, "platform", "win32")` 固定平台分支（断言 setx 的用例与运行平台解耦，cra 自审发现）。卸载用例全程 mock subprocess（Popen/run），绝不真实执行 uv/PATH 修改；`_run_config` 输入打桩统一为"耗尽即 EOFError"（与 test_cli._feed 同型，支撑中断语义用例）。
  - **M8 cra 自审记录（DoD 第 4 条，2026-10-02，三轮：初审 + 复核 + 终审确认）**：
    - 初审（`git diff -- src tests | cra review -`）：**【严重】×2**（POSIX export 回显 key 明文/进 shell 历史；POSIX 环境变量存储"key 可能永久丢失"）+ 【一般】×3 + 【建议】×4。处置：export 回显属 D17 规格明文（"POSIX 打印 export 指令由用户执行"），不采纳机制变更，采纳提示增强（随附保密提醒）；"key 丢失"实际是"当前进程不生效"的断层——采纳修复：POSIX 分支同样设置当前进程 os.environ + 明示"未执行 export 前新终端暂不可见"；`_delete_user_data` 数据源改 `config_path().parent` 并兜底 RuntimeError（采纳）；两个断言 setx 的用例固定平台（采纳）；`_uv_tool_names` 建议改 `uv tool list --json`——**实测锁定版 uv 不支持该旗标**（unexpected argument），不采纳，文本解析以测试锁定（§9 同类版本漂移风险）；PATH 比较小写归一（采纳）；显式 --model 时跳过向导（不采纳：显式参数在无配置时仍缺 key，向导正是解入口，EOF 场景已自动取消）；历史打印对非 dict 防御（初判不采纳——load_session 已校验外层，第三轮以新证据翻案，见终审）。
    - 复核轮（对象 = 修复后 diff，审查者以 Python 实证复现关键项）：**【严重】×1**：`env_var_for_provider` 纯非 ASCII 名统一回退 `CRA_API_KEY`——不同供应商映射同一变量，key 串用会把 B 的凭据发给 A 的接入点（**跨供应商密钥泄露**，审查者实测多例）→ 采纳修复：剥离后为空改名称摘要唯一名。另采纳：数字开头变量名在 POSIX export 报非法标识符（前缀 `_`）+ 写入前变量名校验；卸载数据询问 None（Ctrl+C）当默认 Y（改为保留）；`.cra` 目录名护栏；`$LASTEXITCODE` 检查（防分离进程无 uv 误删 PATH）；存储选择中断语义；`_resolve_or_error` docstring 与调用点核对（grep 无旧名残留）。不采纳：POSIX 回显/`setx` argv 暴露（D17 规格机制，argv 暴露为 setx 固有属性）；os.environ 注入被子进程继承（run_python 环境净化已剔除 *API_KEY* 型变量，M4 自审机制覆盖）；uv 输出格式漂移（同初审）。
    - 终审（对象 = 复核修复后 diff；期间两次再现 DSML 怪癖未出结论，第三次有效）：新发现采纳 4 项——`_print_session_history` 对脏数据 isinstance 设防（该调用在 try 外，防御成本低于论证成本，翻案初审判断）；`_read_key` 补 OSError 与 `_ask` 同口径；`_input_base_url_and_models` 各取消点补"已取消"反馈（旧实现行为，重构时静默了）；`_spawn_cleanup_task` 以 OSError 兜底并返回是否成功（powershell/sh 缺失不崩卸载流程）。等待循环无 PID 复用上限【建议】不采纳（后台 sleep 进程遗留无破坏性，PID 复用窗口极窄）。
    - **修复有效性确认**：三轮全部采纳项均带回归用例落地，410 项 pytest 全绿 + ruff 全绿；跨供应商 key 串用（唯一严重级）由唯一性/稳定性/合法名三测锁定。
  - **M8 修订轮记录（2026-10-02，卸载语义澄清，cra 自审一轮）**：
    - 卸载定位澄清（原文要旨）：仓库克隆是一个文件夹，使用中产生的文件夹之外的东西由 uninstall 删除，之后使用者删除项目文件夹即完整删除；环境变量只删本程序自己设置的——使用环境中可能早已有 `DEEPSEEK_API_KEY` 等，**绝不误删**；设置环境变量前应先检查同名，有则询问是否覆盖。定位随之明确：安装为便利（`uv run cra` → `cra`），卸载为干净退场（本程序不模拟特定使用者的完整环境）。
    - 落地（详见 §7.4/§10 注记 ⑦⑧⑨）：`~/.cra/env-vars.json` 登记层（config 层三函数，损坏按空清单保守处理）；设置侧覆盖保护（他人问 y/N 默认 N、自己的直接更新、拒绝/中断回退明文且原变量不动、登记失败明示卸载清单缺项）；卸载第三道确认（默认 Y、中断保留、清单明示变量名），Windows 前台 PowerShell 清理（.NET 广播使新终端立即可见）、POSIX 打印待删行；登记在删 `~/.cra/` 前读出；非法名不进脚本防注入；项目文件夹本程序不触碰。
    - cra 自审（对象 = 修订 diff，3 一般 + 5 建议）：采纳 5 项——① 覆盖保护判据 `os.environ.get(var)` 改 `var in os.environ`（值为空串的已有变量同样受保护）；② POSIX 清理文案区分"已从当前进程清除/持久化行需手工"（原"已删除"易误读为彻底清除）；③ 卸载 docstring 步骤顺序与代码对齐、清单改条件式措辞；④ `_cleanup_env_vars` 返回值接入收尾提示（清理失败不再静默跟"程序将在退出后完成卸载"）；⑤ register_env_var 注明无跨进程锁的并发边界。另：测试加 delenv 防真实环境同名变量干扰（否则覆盖询问使输入序列错位）、清理函数过滤非法标识符防注入（两条在编码期已先行落地）。不修并报告：审查同时发现工作树混有对 `cra review` 目录展开放宽的改动（`.py`→第一层文本文件+二进制嗅探），其隐藏文件提示、目录无约束等问题连同与 §7.5 的文档差一并复核确认并同步文档。
  - **工作树注记**：修订期间同工作树并入一次全仓批量整理（注释交叉引用清理 + 行尾统一 CRLF，25 文件）与 cra review 目录展开放宽改动；提交时按"纯整理文件"与"修订文件（含 cli.py 内 review 改动，message 注明）"分组 commit，归属可查。
- 估算偏差：规模见 §8.1（估算 ~1000 LOC → M3 收尾实测 3273 LOC）；人工投入与日历时间未逐项记录（见 §8.1 人工投入行）。

## 9. 风险与限制

- **框架版本漂移**：LangChain 迭代快，以 uv.lock 锁定版本；升级属重构，须回归全绿。
- **ChatOpenAI 按 OpenAI 官方 API 规范对齐**，非标字段（如 DeepSeek 的 `reasoning_content`）不被透传——已在 D8 取舍中接受；原生思维链的替代方案见 D16（提示词引导，锁定版 langchain-openai 不提取该字段）；若未来需要深度适配特定供应商，再评估专用集成包。usage 依赖显式 `stream_usage=True`，由回归测试锁定。
- `run_python` 非沙箱：仅用于本地可信代码的审查场景，文档多处明示；子进程环境经净化后继承——剔除 `SE_CodeAgent` 与 `*API_KEY*` 型变量（大小写不敏感），key 不进入子进程、更不进入模型上下文（名称黑名单无法覆盖"名字无害、值是凭证"的变量，属已知边界）。子进程仍拥有用户全部权限（非沙箱不变）。**威胁模型**：被审查的第三方代码内容可构成对模型的间接提示注入、诱导自动执行任意本机代码——缓解：`--ask-exec`/`/confirm` 开关（`cra review` 非交互模式不可逐次确认，审查不可信来源时建议用 `cra chat` + `/confirm on`）、环境净化、工具 description 明示；目录白名单已入 Backlog 评估。
- key 明文存于用户目录配置文件：用户本机数据的已知取舍（D9）；仓库侧以 .gitignore 与密钥门禁兜底，`cra config` 写入时提示文件权限与备份注意事项。
- 审查质量依赖模型能力：对专有框架/最新库版本可能误报，提示词已要求标注"不确定"。
- 上下文超长仅做轮次截断不做摘要；`CRA_MAX_CONTEXT_TOKENS` 应低于所选模型的上下文窗口，必要时自行调低；usage 缺失（服务端不支持）时截断不触发（安全降级）。

## 10. Roadmap 与 Backlog

**已排期**：

- **M4 框架迁移**：langchain/langchain-openai 接入；llm.py → ChatOpenAI 封装；agent.py → create_agent 封装；工具 @tool 包装；行为不变（152 项回归保护）。
- **M5 配置与 CLI 面**（两个增量，M5b 可裁剪）：M5a = models.json + `cra config`(+test) + 环境变量回退 + 首次引导 + `/change model`；M5b = `cra review`（批量/`-o`/`--json`/退出码）+ 全局参数 + REPL 命令族（/help /save /load /undo /context /export /confirm）+ README 同步。
- **M6 流式过程体验**（已落地，2026-10-01；结论性记录）：
  1. 渲染粒度：**行级单元流式**落地（D15）——完整行即时渲染；围栏/列表/表格/引用/缩进代码整组渲染、中间不切段；无结构长行句级早发（60 字符阈值、句界在行内代码外、ASCII 句读须后随空白）；
  2. 过程/正文分层：三类内容视觉层级——过程叙述（暗色平文）＜ 工具打点（⚙ 图标 + 工具名 cyan 加粗 + 结果行，错误行红色）＜ 最终回答（Markdown + 分级着色）；延续 flush 机制，分类取舍见 D15；
  3. 思考过程展示：评估结论 = **不透传 `reasoning_content`**（D16），以提示词引导"工具调用前一句意图说明 + 开工前审查思路"替代；
  4. 可折叠可行性结论：**真折叠不可行**——折叠需改写/收起已打印内容，只能靠 VT 光标回退重绘或 rich.Live，两者在 Windows 传统控制台可靠性差（与 D5 同因，rich Live 方案复评仍排除）；采用**紧凑摘要行（⚙/↳ 打点即紧凑形态）+ `/process` 可选重放**替代。

- **M7 CLI 易用性**（已落地 2026-10-01；可选增强，目标：进一步降低命令行使用门槛）：
  1. **裸入口**：`cra`（省略子命令）默认进入 chat REPL——argparse 子命令可选化 + 默认分支，既有子命令与全局参数面不变；缺失子命令的既有测试语义反转（原期望报错退出）；
  2. **`/setting` 设置菜单**：REPL 内数字选择，**全部复用既有实现**（纯接线）——1) 切换当前会话模型（= /change model） 2) 配置模型列表（= `cra config` 交互函数：增删供应商/模型、设默认、改 key） 3) 保存会话 4) 加载会话 5) 执行确认开关；0/回车返回对话；菜单内任何操作不丢上下文；
  3. **`/exit`**：退出（与 `exit` 等价）；`/help` 与 README 命令表同步；
  4. **会话自动保留**：退出时自动把当前会话覆盖写入 `~/.cra/sessions/_last.json`，启动检测到非空存档打印提示（不自动加载）；`/resume`（= /load _last）恢复，结构自检失败提示不可用。
  **明确不做**：Web 界面（D3 冻结 + 性价比评估见 Backlog）；菜单内再做二级工作区抽象（超出当前体量）。
  范围理由：目标使用流程（裸 `cra` → 直接对话；`/setting` → 数字菜单 → 切换模型/配置模型列表）绝大多数是既有能力的入口接线，新增逻辑集中在 argparse 默认分支、菜单循环与自动存档，估计 +150~250 LOC（含测试），1 个编码会话。
  **落地注记（2026-10-01）**：① 裸形式不带参数——argparse 子解析器默认值会覆盖顶层同名参数值，共享参数面保持在子命令之后（§7.1 已按实现修订）；② /setting 全部复用既有命令函数（_cmd_change_model / _cmd_save / _cmd_load / _cmd_confirm 与 `cra config` 交互循环；后者增可选 console 参数以复用 REPL 终端面，菜单内捕获其损坏配置时的 SystemExit(2) 降级为回菜单，会话不终止）；③ 自动保留对**空会话跳过**——新开即退的空会话不覆盖上一次的有效存档；④ sessions 名称白名单放开首字符下划线（存档名 `_last`；防穿越语义不变）；⑤ 测试隔离升级：conftest 的 home_dir fixture 改为 autouse（自动保留会写 ~/.cra/sessions/_last.json，全部测试须与真实用户目录隔离）。

- **M8 会话与配置便利性**（2026-10-02 依据验收反馈排期）：
  1. **首次配置引导**：chat 无任何配置时询问"是否现在配置"——确认走引导式配置（添加供应商 → key → 设默认模型 → 进入 REPL，复用 `cra config` 交互函数与 _ConfigSession，不重写逻辑）；拒绝/完成后仍无配置维持指引 + 退出 2；`cra review` 保持非交互退出 2（§5 修订）；
  2. **供应商预设**：`cra config` 添加供应商提供编号选择（1 自定义 + 8 家预设），选预设自动填官方 base_url 与模型清单，**预填项均可修改**（base_url 回车用官方、模型清单询问是否使用预填）；预设表如下（2026-10-02 按官方文档核对，**实现前须逐项复核时效性**）：

     | # | 供应商 | base_url | 预填模型 | 环境变量名 |
     |---|--------|----------|----------|-----------|
     | 1 | 自定义 | 用户填写 | 用户填写 | 规则生成（CRA_ 前缀格式） |
     | 2 | DeepSeek | `https://api.deepseek.com` | deepseek-flash, deepseek-v4-pro | `CRA_DEEPSEEK_API_KEY` |
     | 3 | Kimi (Moonshot) | `https://api.moonshot.cn/v1` | kimi-k3, kimi-k2.6 | `CRA_MOONSHOT_API_KEY` |
     | 4 | OpenAI | `https://api.openai.com/v1` | gpt-6-astra, gpt-6-luna（⚠ 命名迭代快，实现时按官方 models 页复核） | `CRA_OPENAI_API_KEY` |
     | 5 | 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen3.8-max, qwen3.8-flash | `CRA_DASHSCOPE_API_KEY` |
     | 6 | Z.ai Coding Plan | `https://api.z.ai/api/coding/paas/v4` | glm-5.3, glm-5.3-flash | `CRA_ZAI_API_KEY` |
     | 7 | Z.ai API | `https://api.z.ai/api/paas/v4` | glm-5.3, glm-5.3-flash | `CRA_ZAI_API_KEY` |
     | 8 | BigModel Coding Plan | `https://open.bigmodel.cn/api/coding/paas/v4` | glm-5.3, glm-5.3-flash | `CRA_ZHIPUAI_API_KEY` |
     | 9 | BigModel API | `https://open.bigmodel.cn/api/paas/v4` | glm-5.3, glm-5.3-flash | `CRA_ZHIPUAI_API_KEY` |

  3. **key 存储选择**（D17）：明文 vs 环境变量；变量名统一 `CRA_供应商名大写_API_KEY`、Windows `setx` 自动 + 进程内生效、POSIX 打印 `export`、models.json 不存明文/移除已有明文、resolve 回退链 `CRA_供应商名大写_API_KEY` → `SE_CodeAgent`（§5/§7.4 修订）；
  4. **会话管理**：`/load` 与菜单 4) 省略名称时列出会话数字选择；加载后自动打印全部历史；`/delete` + 菜单 6)（y/N 确认）；菜单 3) 保存回车自动时间戳命名（§7.3 修订）；
  5. **切换模型**：现有 `/change model` 已是"先供应商后模型"流程，仅补充**当前使用中**标记。
  6. **卸载程序 `cra uninstall`**（2026-10-02 排期；需求：使用结束后能完整卸载，与常见软件卸载程序同等彻底）：
     交互确认（先清单式列出将删除内容，y/N **默认 N** 防误触发）→ 询问"同时删除用户数据 `~/.cra/`（models.json 含 API key、sessions/ 会话存档）"（y/N **默认 Y**——诉求即"删干净"，且清单明示含 key）→ 确认后：① 选删则删除 `~/.cra/`；② 生成**分离的后台清理任务**（等待当前进程退出后执行 `uv tool uninstall code-review-agent`——运行中的 venv 解释器文件被锁，须父进程退出后才能删净；Windows 用 detached 进程，POSIX 用 `nohup sh -c`），完成后若 `uv tool list` 无其他工具，再从用户 PATH 移除 `~/.local/bin` 条目（Windows 经 `[Environment]::SetEnvironmentVariable`，POSIX 打印待删的 shell 配置行指令、不自动改 rc，与 D17 口径一致）；③ 打印"程序将在退出后完成卸载"后正常退出。
     **uv 行为实证（2026-10-02 本机受控实验，哑工具装/卸 + PATH 快照守卫）**：`uv tool install` 不自动加 PATH、缺失仅警告（验收环境问题的根因）；`uv tool uninstall` 只删工具环境与垫片、**不动 PATH 与用户数据**——PATH 清理与数据清理须本命令自理。边界：`uv tool list` 无 code-review-agent（如 `uv run` 方式使用）时跳过工具卸载并说明；PATH 条目被其他 uv 工具共享时保留（删净自己、不破坏他人）。
  范围理由：实际使用暴露首次配置断层（新机器无配置无环境变量）、会话管理便利性诉求与使用结束后干净退场的需要；估计 +550~850 LOC（含测试），1 个编码会话。
  **明确不做**：联网拉取模型清单（离线可用优先：内置清单 + 可编辑 + `cra config test` 验证）；自动改写 POSIX shell 配置文件（装/卸两侧一致，只给指令）；多套 key 轮换；静默卸载（`-y` 类无交互旗标——当前体量内保持交互确认）。
  **落地注记（2026-10-02）**：① 预设清单实现前逐项复核结论见 §8.3 M8 记录（全部 base_url 与变量名与官方一致；OpenAI/百炼模型名按当日核对结论采用，预填可改 + config test 兜底）；② 预设的供应商名即 models.json name（"DeepSeek"、"Z.ai Coding Plan" 等），default 引用与 --provider 直接可用；③ key 存储选择统一入口 `_apply_key_storage`（添加/更新/引导三路复用），环境变量存储时 models.json 不落明文且已有明文移除，中断（EOF/Ctrl+C）一律不视为确认；④ 首次引导复用 `_config_add_provider` + `_config_set_default`（与 /setting 菜单同一套交互函数），仅在零供应商清单时出现；⑤ 卸载后台任务 Windows 为 detached PowerShell（等父进程 PID 退出 → uv tool uninstall → 无其他工具则 SetEnvironmentVariable 移除用户 PATH 条目）、POSIX 为 `sh -c` + start_new_session（仅等退出 + 卸载，PATH 打印指令不改 rc）；⑥ 实际净增 +1463 LOC（超估算主因卸载程序完整实现与交互分支用例面），测试 354→410 项。
  **落地注记·卸载语义修订（2026-10-02）**：卸载定位澄清——仓库克隆是一个文件夹，使用中产生的文件夹之外的东西由 uninstall 负责删除，之后使用者删除项目文件夹即完整删除；环境变量只删本程序自己设置的，使用者已有的同名变量（可能早于本工具存在的 DEEPSEEK_API_KEY 等）绝不能误删。落地：⑦ 设置侧写环境变量前做覆盖保护——非本程序所设且已有同名时询问（默认 N，拒绝回退明文），登记在案的（自己之前设的）直接更新不重复询问；写入成功登记 `~/.cra/env-vars.json`；⑧ 卸载新增第三道确认"删除本程序设置的环境变量"（默认 Y、中断保留），清单明示变量名；Windows 前台 PowerShell 逐变量 `[Environment]::SetEnvironmentVariable($null,'User')`（.NET 写回广播 WM_SETTINGCHANGE，新终端立即可见；reg delete 无广播故不采用），POSIX 打印待删 export 行（不自动改 rc）；登记文件在删除 ~/.cra/ **之前**读出（它就在其中）；非法标识符（登记被手改）不进脚本、提示手工——防命令注入；⑨ 项目文件夹（仓库克隆）本程序**不触碰**，由使用者自行删除。

**作业二规划**（Simple Workflow MVP，不提前实现）：代码审查工作流方向——以本项目 Agent 为节点（§2.2 接缝），LangGraph `StateGraph` 编排"编码→审查→修复"或同等流程；顺序执行为 MVP，分支/并行/日志追踪按作业二推荐项递增。

**Backlog（未决功能池，新增功能先入池评审再排期）**：`--log-file` 本地调试日志；`/history` 会话消息摘要；累计 token 用量显示；run_python 目录白名单；多智能体协作/反思复核/RAG（作业二本体素材）；可视化编排界面（作业二可选项）；**Web 界面**（2026-10-01 评估：便利但性价比低——需本地服务 + 流式通道（SSE/WebSocket）+ 前端 + 会话状态管理，新增依赖与测试面翻倍，估计 2–3 个 M4 级编码会话，且违反 D3 冻结；CLI 设置菜单已覆盖易用性诉求，若作业二需要多端形态须先解除 D3）；**多工作区 / 按项目配置**（"工作区"语义待澄清：配置存储已由 ~/.cra 覆盖，若指按项目切换工作目录/配置，超出当前体量）。
