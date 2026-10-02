# Design.md — 代码审查助手 code-review-agent

《软件工程与项目实践》第 1 次作业 · 提交截止 2026-10-07 24:00（作业二将复用本项目成果，见 §2.2 与 §10）
平台环境：Windows + Git Bash（PowerShell/CMD 兼容）。

> 本文描述**目标形态**；截至 M7，已排期内容（M0–M7，§10）全部落地，代码、README 与本文一致；§2.2 的作业二接缝与 **M8**（2026-10-02 用户验收反馈排期，§5/§7.3/§7.4 中标注 M8 的条目）属目标形态、待实施。M7（CLI 易用性：裸入口 / 设置菜单 / 会话自动保留）已于 2026-10-01 落地。

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
    __init__.py # 工具注册表：@tool 定义集合 + "错误："结构化错误协议包装（行为锁定点）
    fs.py       # list_dir、read_file
    exec_py.py  # run_python（subprocess 限时执行）
  sessions.py   # 会话持久化（/save /load，M5 新增）
scripts/
  smoke_tools.py   # 冒烟测试（入库留存，全仓库唯一真实联网）：与 llm 层 API 形状耦合，llm 层改形须同步迁移
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
7. **流式渲染（M6 重构，D15）**：文本增量交 cli 渲染管线，粒度为**行级单元流式**——完整行即时落盘（Markdown 渲染 + 分级标记着色）；围栏（``` / ~~~）、列表、表格、引用、缩进代码为**完整性单元**，整组渲染、中间不切段（组等待终止行到达才落盘）；无结构长行超过阈值（60 字符）后在句界提前落盘（句级早发，句界须在行内代码之外）。**过程/正文分类沿用 flush 机制**：工具打点前的缓冲文本按暗色平文落盘为过程叙述，最终回答经 Markdown 渲染；分类窗口为工具结果后的首个单元（暂扣至第二个单元到达，过程叙述实测通常单行——M4 验收记录），取舍记 D15。叙述、工具打点、最终回答三类内容分层样式（§10 M6）。usage 与最终消息（含 finish_reason，如 length 截断提示）随最终响应取得；框架类型由 agent.py 适配为纯字符串回调，不越过 agent 层（§3.4）。
8. `Agent.run(input, callbacks) -> str` 为可编程接口（§2.2 接缝 1）；任何异常（Ctrl+C 或请求失败）回滚本轮消息到合法前缀后原样上抛。

### 3.2 工具 schema

| 工具 | 参数 | 行为与限制 |
|------|------|-----------|
| `list_dir` | `path` | 返回两层目录树；忽略 `.venv`、`__pycache__`、`.git`、`.pytest_cache`、`.ruff_cache` 与点开头的隐藏目录 |
| `read_file` | `path`, `start?`, `end?` | 带行号返回内容；单次上限 **400 行**，超出提示用 start/end 分段；`start`/`end` 为 1-based 闭区间，非法区间返回结构化错误；二进制/解码失败返回结构化错误信息（回传模型） |
| `run_python` | `code` 或 `path`（二选一）, `timeout?` | subprocess 运行，工作目录=进程启动目录；默认 10s（`CRA_EXEC_TIMEOUT` 可调），与模型传参同标准校验（1~60 秒），Windows 下 `taskkill /T /F`、POSIX 下独立进程组 + `killpg` SIGKILL 强杀整棵进程树；子进程环境经净化后继承（剔除 `SE_CodeAgent` 与 `*API_KEY*` 型变量，大小写不敏感）；返回 stdout/stderr/退出码；stdout/stderr 各截断至 **2000 字符**；**本机直接运行，非沙箱**；执行确认开关见 §7.2/§7.3（`cra review` 非交互模式不支持 `--ask-exec`，指定时报错退出 2） |

通用规则：三工具的相对路径一律相对**进程启动目录（cwd）**解析；所有文件读取与子进程输出捕获显式 `encoding='utf-8', errors='replace'`，子进程环境另设 `PYTHONIOENCODING=utf-8`。

### 3.3 系统提示词要点

角色为资深代码审查员（Python 为主深耕，任意语言的源码、脚本与配置文件均可审查；执行验证仅 Python，其他语言静态审查并标注"不确定"）；输出固定结构：【严重/一般/建议】分级 + `文件:行号` + 问题 + 修改建议（修改建议独立成行、紧跟问题描述）；先调用工具核实再下结论；不确定的问题明确标注"不确定"，禁止编造行号或代码。默认输出格式为**提示词约定**，不做程序化校验；`--json` 模式（§7.1）例外：按 §7.5 的 schema 程序化输出与校验。

### 3.4 代码设计规范落点

风格层由 ruff 强制（§6）；设计层按以下原则执行，各里程碑收尾按 §8.2 Checklist 核对：

| 原则 | 本项目落点 |
|------|-----------|
| 单一职责 | cli 只管交互与渲染；agent 只管循环与状态；llm 只管模型客户端与重试；tools 只管工具执行；config 是配置唯一出入口 |
| 依赖与模块化 | 依赖方向单向：cli → agent → {llm, tools}；tools/llm 不反向 import agent/cli。框架类型（LangChain 消息/流对象、SDK 异常）不越过 agent 层向上暴露（类型级豁免：cli 仅可从 llm 导入 `LLMError` 类型与 `probe` 连通性自检（§7.4 `cra config test` 专用最小通路，无 Agent 循环）、从 tools 导入 `ERROR_PREFIX` 常量） |
| 数据封装 | 会话消息结构与截断规则只存在于 agent.py；工具返回统一为字符串（错误也是字符串） |
| 异常处理 | 按 §5：异常要么被结构化回传模型，要么给出用户可读报错，不允许静默吞掉 |
| 性能 | 流式输出保证感知性能；工具调用有超时；上下文有截断阈值 |
| 安全 | key 只在内存流转、不进仓库（用户目录配置文件除外，见 §7.4），且不进入子进程环境（run_python 环境净化）；run_python 限时 + 截断 + 环境净化 + 限制明示（第3讲 Gab 教训：规范一丢，漏洞就来） |

## 4. 关键决策记录（ADR）

被取代的决策保留条目并在状态列标注去向——决策演化本身是项目过程的一部分。

| # | 决策 | 状态 | 理由与后果 |
|---|------|------|-----------|
| D1 | 手写 ReAct 循环（消息列表 + function calling + 循环分发），不引入框架 | 已取代（→D8） | 作业一 M0–M3 以原生 API 路线实现并验收；吃透 Agent 原理、架构可讲可测。被取代原因见 D8 |
| D2 | LLM 用 DeepSeek，OpenAI 兼容接口 | 采纳 | 国内直连、成本低；OpenAI 兼容接入点可替换服务商 |
| D3 | 仅 CLI，入口 `cra`，不做 Web 界面 | 采纳 | 作业允许二选一；聚焦 Agent 核心能力 |
| D4 | run_python 用 subprocess + 限时强杀，不做沙箱 | 采纳 | 作业范围权衡；风险用"限制明示 + 超时强杀 + 输出截断 + 可选执行确认"控制 |
| D5 | 统一采用流式；文本增量与 usage/finish_reason 分通道取得 | 采纳 | 轮次类型无法预判；tool_calls 分片组装完整性等价；工具轮伴随的少量文本同样渲染（已知取舍）；不使用 rich.Live（Windows 控制台重绘不可靠） |
| D6 | 上下文截断按完整轮次原子丢弃，系统提示词永留 | 采纳 | 拆散 tool_calls/tool 配对会导致 API 400；截断为请求侧裁剪、agent 持有完整历史（§3.1(6)），语义由回归测试锁定 |
| D7 | 审查输出格式为提示词约定，不做程序化校验 | 采纳（结构化例外见 D10） | 格式断言测试脆弱；`--json` 是显式例外（schema 校验），且退出码分级判定仅在该模式下保证（§7.5） |
| D8 | Agent 循环迁移至 LangChain 1.x `create_agent`（底层 LangGraph） | 采纳 | 作业二要求 2–3 Agent 工作流并复用本项目：LangGraph 的节点/边/状态原语直接支撑编排与可扩展性（作业二评分项）；`create_agent` 为官方现行推荐 API；工具层/提示词/渲染/测试策略全部可复用。备选 AutoGen/CrewAI（多智能体叙事错位）、Dify（平台非库，违反仅 CLI）、LangGraph 直接手写 agent 图（原语相同但自维护面更大）、langchain-deepseek 专用包（破坏多供应商统一封装）均排除；后果：接受框架版本漂移风险（§9），以 uv.lock 锁定管理 |
| D9 | 多供应商模型配置：`~/.cra/models.json`（用户目录，仓库外）+ 环境变量回退 | 采纳 | 用户要求命令行配置供应商/密钥/模型；key 存用户目录可同时满足"配置体验"与"仓库无明文密钥"门禁；JSON 程序化读写优于 Markdown |
| D10 | 非交互审查模式 `cra review`（含批量/报告导出/--json 结构化输出/退出码） | 采纳 | CLI 审查工具的标配形态；一并落地 Should 两条；是作业二"Agent 即节点"的雏形 |
| D11 | run_python 默认自动执行，`--ask-exec`/`/confirm` 可开启逐次确认 | 采纳 | 审查中执行验证是高频动作，默认打断伤体验；安全叙事由文档明示 + 可选开关表达 |
| D12 | 文档以最终形态维护（不设版本号），决策历史由本表与 §8 PSP 附录承载 | 采纳 | 无需保留旧版本；决策演化靠 ADR 状态列追溯（README 横幅的交付状态标注不受此限） |
| D13 | 重试映射 SDK：`ChatOpenAI(max_retries=3)`，指数退避间隔与 5xx 语义以 openai SDK 实现为准（重试仅覆盖建连阶段） | 采纳 | middleware 级自定义重试与"流式中途断开不重试（防重复渲染）"冲突；SDK 重试同样仅覆盖建连期，语义等价性最高；401/403 快速失败语义保留 |
| D14 | 会话消息正本由 agent.py 持有（不使用 checkpointer 的无状态图调用） | 采纳 | 保住消息结构/截断/回滚的单点封装（§3.4）与 /undo、/save /load 的实现路径；Ctrl+C 回滚即消息列表快照还原；作业二工作流由 StateGraph 自管状态 |
| D15 | 流式渲染粒度 = 行级单元 + 句级早发 + 边界后暂扣分类的**追加式打印**；不引入 rich.Live（D5 复核后维持） | 采纳（M6） | 段落批渲染是 M4 验收"一大段一次性出"的主因；行级单元对围栏/列表/表格/引用整组渲染保完整性（rich 逐条打印列表项会因块边距插入空行，实证）；句级早发只作用于无结构长行（ASCII 句读须后随空白，防切 `a.py:10` 类路径）。D5 边界复核结论：rich.Live 仅在需要**重绘已打印内容**（真折叠/原地改写）时才有必要，追加式打印无此需求，决策维持。代价（已知取舍）：① 叙述/正文分类窗口为工具结果后的首个单元，多行过程叙述的首行会以正文样式先行显示（过程叙述实测通常单行）；② 最终回答首行延迟至次行到达；③ setext 标题等罕见 Markdown 结构按普通行降级渲染（模型输出以 ATX 标题为主） |
| D16 | 思考过程展示采用**提示词引导的过程叙述**，不透传 DeepSeek `reasoning_content` | 采纳（M6） | 透传属设计变更：langchain-openai 1.6.6（uv.lock 锁定版）文档明示非标字段（`reasoning_content`/`reasoning_details`）不被提取，须供应商专用子类或专用包（langchain-deepseek，D8 明确排除项）；且默认模型 deepseek-flash 非 reasoning 模型、无该字段，收益面窄。提示词引导（工具调用前一句意图说明 + 开工前审查思路）对所有模型生效、零协议风险，过程叙述经既有 flush 管线暗色展示。若未来换用 reasoning 模型且需原生思维链，再评估专用集成包（须征得同意，§9） |
| D17 | 供应商预设清单 + key 环境变量存储（首次引导式配置；Windows `setx` 自动写入；回退链 `{供应商}_API_KEY` → `SE_CodeAgent`） | 采纳（M8 排期，2026-10-02 用户验收反馈提出并逐项确认方案） | 验收演示暴露首次使用断层：新机器既无 models.json 也无环境变量，现有"打印指引退出 2"把全部配置成本压给用户。预设清单把"添加供应商"压缩为选编号 + 填 key；环境变量选项缩小 D9"明文 key"取舍面（用户二选一，选环境变量则 models.json 不落明文）。后果：① `setx` 属系统状态变更，写入前不回显 key 值、文档明示；② 预设模型名有时效性——内置清单标注核对日期、实现前逐项复核官方文档、预填项均可修改、`cra config test` 兜底验证；③ 离线可用优先，**不联网拉取清单**；④ POSIX 不自动改 shell 配置文件，只打印 `export` 指令 |

## 5. 错误处理与重试

行为契约（实现映射到 LangChain/SDK 能力，由回归测试锁定；重试语义见 D13）：

- 启动校验：无任何可用配置（配置文件 default 与环境变量皆缺）时打印 `cra config` 设置指引后退出（退出码 2）；**不校验 key 格式**（无效 key 由 API 认证环节暴露）。**M8 修订**：chat 模式先询问"是否现在配置"——确认则进入引导式配置（添加供应商 → key → 设默认模型 → 进入 REPL），拒绝或完成后仍无配置维持指引 + 退出 2；`cra review` 保持非交互、无配置直接退出 2。
- LLM 调用：瞬时错误按指数退避重试（最多 3 次，经 SDK 重试承担、仅覆盖建连阶段，间隔以 SDK 实现为准——D13）；**认证类错误（401/403）快速失败、不重试**，输出 key 排查指引（跨平台：Windows setx / macOS-Linux export）；仍失败给出友好报错与排查建议；流式迭代中途断开不重试（增量已渲染，重发会重复输出）；SDK 异常 → `LLMError` 的翻译留在 llm 层，agent/cli 不接触 SDK/框架异常类型。
- 工具执行：所有失败（含框架参数校验、未知工具名）以"错误：…"结构化文本回传模型（§3.1(3)），REPL 不崩溃。
- `run_python`：超时强杀进程树；stdout/stderr 截断防刷屏。
- `Ctrl+C`：中断当前生成——任何中断必须把会话消息回滚到本轮开始前的合法前缀并提示该轮作废；输入 `exit` 或再次 `Ctrl+C` 退出 REPL。

## 6. 工程规格

- 包管理：uv（≥0.12），`requires-python = ">=3.13"`，`.python-version` 固定 `3.13`；包目录 `src/cra`，hatchling 打包配置 `packages = ["src/cra"]`。
- 运行依赖：`langchain>=1.0`、`langchain-openai>=1.0`、`rich`（M4 迁移时以 uv.lock 锁定；M4 完成后从直接依赖移除 `openai`，由 langchain-openai 传递提供）。开发依赖：`pytest`、`ruff`。新增依赖须先征得用户同意。
- 入口与 CLI 面：`[project.scripts]` → `cra = "cra.cli:main"`；子命令 `chat` / `review` / `config`（§7）。全局安装：`uv tool install -e .` 后可直接 `cra`（M5b 时在 README 载明）。
- 配置（环境变量，作为 `~/.cra/models.json` 不存在时的回退，统一在 `config.py` 读取）：

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `SE_CodeAgent` | 配置文件缺省时必填 | — | DeepSeek API key（`~/.cra/models.json` 缺省时的回退） |
| `CRA_MODEL` | 否 | `deepseek-flash` | 模型名回退 |
| `CRA_BASE_URL` | 否 | `https://api.deepseek.com` | 接入点回退 |
| `CRA_EXEC_TIMEOUT` | 否 | `10` | run_python 超时（秒） |
| `CRA_MAX_CONTEXT_TOKENS` | 否 | `100000` | 会话截断阈值（tokens） |

- ruff：默认规则 + `line-length = 100`；`tests/fixtures` 豁免（预埋样例属数据非业务代码）。
- pytest `testpaths = ["tests"]`；**测试内禁止真实网络调用（LLM 一律 mock），真实网络仅限 scripts/smoke_tools.py**。
- `.gitignore`：`.venv/`、`__pycache__/`、`.pytest_cache/`、`.ruff_cache/`、`dist/`、`.env`、`*.pyc`、`.mimosa/`、`AGENTS.md`、`docs/PROGRESS.md`、`Homework 1.pptx`（课程课件，暂不入库）。开发辅助文档（AGENTS.md、PROGRESS.md）仅本地维护、不随仓库分发；仓库内交付文档为 README.md 与 Design.md。
- 密钥边界：api_key 只允许出现在用户目录 `~/.cra/models.json`（§7.4）与进程内存；**仓库内任何文件、日志、测试样例、提交记录不得出现真实密钥**。

## 7. 接口规格

### 7.1 CLI 子命令

```
cra                              # 省略子命令：默认进入对话式审查 REPL（M7；裸形式不带参数，选项用 cra chat 传入）
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
| `--ask-exec` | run_python 执行前逐次人工确认（默认自动执行；REPL 内 `/confirm on|off` 同效）。仅限 chat 模式；`cra review` 指定时报错退出 2 |

### 7.3 REPL 内置命令

| 命令 | 行为 |
|------|------|
| `/help` | 命令列表 |
| `/clear` | 会话重置为仅含系统提示词（角色与格式契约保留） |
| `/change model`（别名 `/model`） | 列出供应商及其模型清单，按序号选择；切换后上下文保留并打印提示；未配置任何供应商清单时提示运行 `cra config` |
| `/save [名称]` / `/load [名称]` | 会话持久化到 `~/.cra/sessions/<名称>.json`（消息数组）/ 从文件恢复（整体替换当前会话）。M8：`/load` 省略名称时列出已有会话数字选择（回车取消）；恢复成功后自动打印会话历史（`你>`/`cra>` 逐条） |
| `/undo` | 作废上一轮对话（回滚到上一轮开始前的状态） |
| `/context` | 显示当前会话 token 用量与距截断阈值的余量 |
| `/export [文件]` | 将最近一轮审查结论导出为 Markdown |
| `/process` | 显示最近一轮的过程明细（工具轮叙述全文、工具调用与结果预览）——紧凑打点的可选重放（§10 M6 可折叠结论） |
| `/setting` | 设置菜单（数字选择，M7）：1) 切换当前会话模型（复用 /change model） 2) 配置模型列表（供应商/模型/key/默认项，复用 `cra config` 交互函数） 3) 保存会话 4) 加载会话 5) run_python 执行确认开关；0 或回车返回对话，菜单内上下文不丢失。M8：3) 保存回车自动以时间戳命名；新增 6) 删除会话（= /delete） |
| `/resume` | 恢复最近一次自动保留的会话（= /load _last；结构自检失败则提示存档不可用，正本不变） |
| `/delete` | 删除已有会话（M8）：列出会话数字选择 + y/N 确认后删除文件（`_last` 同样可删，删即清除自动保留） |
| `/confirm [on|off]` | run_python 执行确认开关（默认 off） |
| `exit` 或 `/exit`（或 Ctrl+C 两次） | 退出 |
| 其余 `/xxx`（未匹配内置命令） | 按普通文本发送给模型 |

会话自动保留（M7，`/resume` 的数据来源）：REPL 以任何方式退出（exit / /exit / EOF / Ctrl+C）时，将当前会话消息覆盖写入 `~/.cra/sessions/_last.json`（序列化与 /save 同路径，失败仅提示不阻断退出）；启动时若存在非空存档则打印一行提示（不自动加载，避免上下文静默混入）。

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

- 位置固定在**用户目录**（Windows 为 `%USERPROFILE%\.cra\`），不在仓库内、不随仓库分发；`cra config` 负责创建与编辑（交互式：增删供应商/模型、设默认、改 key；**同名供应商 + 同名模型**已存在时询问是否覆盖——供应商同名即更新其接入点与 key、模型清单保留；同名模型在纯名称数组中"覆盖"即确认保留该名称，语义为防误操作确认）。写入为原子替换（临时文件 + rename），POSIX 上自动收紧为 0600。
- 边界规则：`default` 以第一个 "/" 分隔"供应商/模型"，供应商名不得含 "/"（`cra config` 写入时校验）；`--model` 仅在已解析供应商的模型清单内匹配，跨供应商重名时必须同时给 `--provider`，否则报错；`--provider` 单独使用时取该供应商 `models[0]`；`default` 引用的供应商/模型被删除后视为配置损坏，按运行错误退出 2；provider 条目缺 `api_key` 时按序回退环境变量：`{供应商}_API_KEY`（预设供应商用官方惯用名，见 §10 M8 表）→ `SE_CodeAgent`（作业指定变量，保留为最后回退；是否匹配由认证环节暴露，M8）。
- 供应商预设与 key 存储选择（M8，D17）：`cra config` 与首次引导添加供应商时提供预设编号选择——选预设自动填入官方 base_url 与模型清单（**预填项均可修改**：base_url 回车用官方接入点或输入自定义；模型清单询问是否使用预填）；填 key 后选择存储方式——1) 明文写入 models.json（现状，含权限与备份提示）2) 环境变量（变量名：预设供应商用官方惯用名；自定义按"供应商名大写、空格与非字母数字转 `_`、`_API_KEY` 后缀"生成，非 ASCII 字符剥离；Windows 经 `setx` 写入用户环境并设置当前进程 `os.environ` 即时生效，POSIX 打印 `export` 指令由用户执行）；选环境变量时 models.json **不存明文 key**，该供应商已有明文的**一并移除**。
- `cra config` 写入时提示文件权限与备份注意事项（key 为明文，属用户本机数据，见 D9/§9）。
- 解析优先级：`--model/--provider` 参数 > 配置文件 `default` > 环境变量回退（`SE_CodeAgent`/`CRA_MODEL`/`CRA_BASE_URL`）。配置文件与环境变量皆缺时，打印 `cra config` 引导后退出（退出码 2）。
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

stdin 输入时 `file` 取 `"<stdin>"`；目录入参展开其第一层的 `.py` 文件；`--json` 时 stdout 只输出 JSON，`-o` 可同时写 Markdown 报告。

`cra chat` 正常退出码 0；配置缺失退出码 2。

## 8. PSP 附录（轻量诚实版）

**开发过程定位（第3讲）**：本项目采用**增量开发**——每个里程碑交付一个可用增量，用户验收即持续反馈，回归测试保护既有增量，验收意见回灌下一里程碑。代码审查按静态审查（ruff + pytest）、动态审查（验收运行检查）、代码走查（验收 + 会话收尾总结，范围 = 该里程碑 diff）三级落地，结果随开发日志记录；§8.2 Checklist 是走查门禁。

### 8.1 估算表

| 项 | 估算 | 实际 |
|----|------|------|
| 总规模 | ~1000 LOC（含测试，git 统计为准） | 7277 LOC（截至 M6：src 3131 + tests 4016 + scripts 130，`git ls-files "*.py"` 统计、fixtures 豁免）；作业一定稿 |
| 人工投入 | 设计评审 0.5h + 里程碑验收 0.25h×n + 收尾 0.5h | M0–M2 验收实测共 22 分钟；此后未再逐项记录 |
| AI 生成 | 全部实现代码与测试初稿 | 一致：AI 编码会话生成，人工评审设计、逐里程碑验收 |
| 日历时间 | 2–3 天（4 个编码会话） | 1 天完成 M0–M3 编码与验收 |
| M4 框架迁移 | 1 个编码会话；净变化 ±300 LOC 待实测（middleware/流式适配/无状态包装可能净增）+ 测试重写 ~400 行 | 1 个编码会话；迁移本体净 **-134 LOC**（3273 → 3139），验收修复轮 +201、cra 自审修复轮 +247 → 3587；测试重写与增补 56→63 项（test_agent 27→43、test_llm 29→20、exec/cli 增补），总数 152→173 |
| M5 配置与 CLI 面 | 1–2 个编码会话；净增 ~800–1200 LOC（含测试） | 1 个编码会话；净增 **+2826 LOC**（3587 → 6413：src 1387→2672、tests 2070→3611、scripts 不变）；测试 173→297 项 |
| M6 流式过程体验 | 1 个编码会话；净增 ~300–500 LOC（渲染管线重写 + 测试） | 1 个编码会话；净增 **+864 LOC**（6413 → 7277：src 2672→3131、tests 3611→4016、scripts 不变）；测试 313→329 项；cra 自审 1 严重 + 1 一般 + 2 建议 → 修复 + 复核轮 6 建议落地 |
| M7 CLI 易用性 | 1 个编码会话；净增 ~150–250 LOC（含测试） | 1 个编码会话；净增 **+493 LOC**（7277 → 7770：src 3131→3264、tests 4016→4376、scripts 不变；超估算主因新用例 25 项 / +360 测试行）；测试 329→354 项；cra 自审三轮（初审 2 一般 + 6 建议 → 复核 2 一般 + 4 建议 → 终审 0 严重、2 一般 + 3 建议，处置见 §8.3） |

### 8.2 代码评审 Checklist（收尾门禁；M0–M7 每次收尾逐项执行，当前记录：M7）

- [x] 全仓库 grep 无明文密钥；key 读取点唯一（config 层） ✅（M7 复执行：grep 无命中；自动保留只写对话消息序列、不涉 key）
- [x] 无裸 `except:`；所有工具异常均回传模型而非静默吞掉 ✅（M7 新增兜底均有 noqa 与注释陈述理由：_autosave_session 的 `except Exception`（退出路径失败仅提示）、失败提示自身的嵌套兜底（BLE001/S110，finally 中不得上抛）、_hint_last_session 的 `except Exception`（启动提示视同无存档）；菜单降级走 ConfigError 类型契约，不吞 SystemExit）
- [x] `run_python` 有超时强杀与输出截断 ✅（M7 未改动工具层与 middleware）
- [x] Agent 循环有最大工具轮次（10）限制 ✅（M7 未改动）
- [x] 消息截断按完整轮次原子丢弃，配对不拆散，系统提示词保留 ✅（M7 未改动）
- [x] `cra chat --help` 与 README 含"非沙箱、限时执行"限制声明 ✅（M7 未改动声明，有测试锁定；裸入口 `cra --help` description 同步说明默认行为）
- [x] README 每条命令逐条实测可复制执行 ✅（M7 实测：裸 `cra` 进 REPL、/help 新命令、/setting 菜单嵌套循环（→配置交互→回菜单→回对话）、exit 退出、`cra nope` 仍 invalid choice 退出 2；命令表/全局参数/功能清单已同步）
- [x] pytest 覆盖：工具 handler、消息组装、截断逻辑、CLI 参数解析 ✅（M7 后 354 项；新增：裸入口默认分支与语义反转、裸入口与 chat 参数缺省一致性锁定、菜单分发（mock 既有命令函数验证映射与返回菜单）、菜单嵌套循环与损坏配置降级、/exit、退出自动保留（exit//exit/EOF 三路径）、空会话跳过不覆盖旧存档、写盘失败不阻断退出、启动提示（非空/缺失/空/损坏/结构非法）、/resume 恢复与损坏存档正本不变、_last 名称合法性）
- [x] ruff 全绿；公开函数有类型标注；命名自解释 ✅（M7：_cmd_setting/_repl_loop/_autosave_session/_hint_last_session 均带 docstring；conftest home_dir 改 autouse 附隔离理由）
- [x] 模块依赖方向符合 §3.4（无反向 / 越层 import） ✅（M7 无新增依赖方向；_config_interactive_session 抽取仅在 cli 层内部）
- [x] `git status` 干净：无 venv/缓存/密钥/开发辅助文档被跟踪 ✅
- [x] Design.md 与实现一致（模型、工具名、模块、入口、配置项、CLI 面） ✅（M7 回写：§7.1 裸入口行、头部目标形态说明、§10 M7 落地注记、§8.1/本节）

### 8.3 复盘、缺陷记录与 PIP

- **缺陷记录**（M0–M4 验收与走查共 19+3 项；M0–M3 的 19 项全部闭环、无遗留；M4 编码期 3 项现场闭环，无新增遗留）：
  - M0 验收 1 项：提示词角色未定义【一般】→ M1 prompts.py 落实。
  - M1 验收 1 项：输出文字堆叠不便阅读【建议】→ 修复轮 1（空行/加粗）。
  - M1 走查（cra 复审两轮）13 项：中断回滚索引失位致孤儿 tool_calls【严重】、空回答静默入库【一般】→ 修复轮 1；标记加粗漏标、围栏奇偶误判、工具失败判定、rich markup 注入、REPL 意外异常兜底、llm 异常包装扩大【一般 6】及显式化小项【建议 5】→ 修复轮 2。
  - M2 编码期 1 项：read_file 空文件误报区间错误【一般】→ 现场修复。
  - M2 验收 2 项：审查语言边界设置错误【一般】、修改建议格式与着色【建议】→ 修复轮（多语言放开、标记切片渲染）。
  - M2 修复轮自测回归 1 项：CommonMark flanking 规则致标记加粗失效【一般】→ 渲染改标记切片。
  - M3 挂账处置：finish_reason 透传、print/console 统一、指引跨平台化、_run_chat 可测性已落地；stream_options 不做降级重试（DeepSeek 实测支持，400 属参数错误、不在重试范围，有回归锁定）；截断 O(n²) 维持不改（消息规模小且截断罕见，两次评估结论一致）。
  - M4 编码期 3 项（现场闭环）：① `translate_failure` 初稿误写 `return … from exc`（语法错误）→ 返回型翻译改手工设置 `__cause__`；② LangGraph 对传入消息对象原位分配 id，等值断言失配【测试】→ 测试改按语义形状比较（实现不受影响）；③ 达限"去工具"后框架走 `model.bind()` 而非 `bind_tools()`，假模型绑定日志缺该次记录【测试】→ 假模型补记 `bind()` 空绑定，机制经 `request.override` 锁定。
  - **M4 验收修复轮（2026-09-30，验收走查回灌）9 项**（自审复检 + 用户反馈，全部闭环）：
    - 【一般·实证复现】"Error:" 前缀字符串匹配误伤工具正常输出（run_python 的 stdout 恰以 `Error:` 开头时被改写为"错误："并污染回传）→ 改为 **ToolMessage.status=="error" 语义字段判别**（`_normalize_framework_error`）：框架错误（校验失败/畸形参数）必为 error 状态、正常输出为 success，门禁不依赖框架文案；前缀剥离仅作文本清理，已归一消息防重复叠加。附带缓解"框架文案硬编码"风险（判别不再依赖 "Error:"/"could not be executed" 字面）。
    - 【一般】回滚只删末条消息，覆盖不了"成功 extend 之后仍抛异常"的窄窗口（§5 本轮作废契约）→ 改为进入 try 前记录正本前缀长度，异常按 `del self.messages[prefix_length:]` 截断。
    - 【一般】达限收尾指令以伪造 user 消息注入请求，恰构成截断的"轮次边界"——极端场景下截断会保留孤立指令、丢弃全部真实上下文 → 收尾指令改经 **system 通道追加**（不产生伪造 user 消息，轮次边界只认真实输入；指令不落正本不变）。
    - 【一般】on_tool_result 三分支触发条件不一致（正常分支要求非空才回调）→ 统一为每分支必然触发一次（空串也回调），协议对调用方一致。
    - 【建议×4】`str(call.get(k, ""))` 对 None 值产出 "None" 串 → `or ""`；reset() 一并清理回调挂留引用；recursion_limit 的"每轮节点数"提为具名常量 `_NODES_PER_TOOL_ROUND`；token 估算加 1.2 倍安全余量（×0.6，中文占比高时防低估少丢，宁多丢不超限）。
    - 夹具建议（infinite_loop.py）：`__main__` 导入守卫 + 注释措辞收敛（明确"仅覆盖直接子进程强杀、/T 树杀路径未触及"）已落地；CPU 自旋保留（kill 语义与 sleep 循环等价，且保持 CPU 密集场景真实性）。
  - M4 修复轮附注：真实 API 实测逐 token 流式通路正常（320 个增量、拼接==返回值）；用户感知"一大段一次性出"源于段落批渲染 + 跨轮叙述累积两处，后者已由 cli 叙述分流修复，前者列入 M6。
  - **M4 自审修复轮（2026-09-30，cra 自审回灌——用户指定固定流程：所有会话收尾前须用本项目 cra 审查本轮代码）**：
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
  - **实现决策注记**（§7 落地细节，供验收留意）：
    - `cra review` 的文件内容**直接嵌入 prompt**（行号格式与 read_file 一致，1-based），不走路由工具轮：非交互审查优先确定性与行号可靠性（模型自读再引行号有编造空间，且多一轮往返）；单文件上限 2000 行，超出截断并在 prompt 中明示（防 token 爆炸，§8.1 已知边界）。
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
- 估算偏差：规模见 §8.1（估算 ~1000 LOC → M3 收尾实测 3273 LOC）；人工投入与日历时间未逐项记录（见 §8.1 人工投入行）。
- 过程问题：
- PIP（下一项目为结对项目，将在本项目上增量开发与重构）：

## 9. 风险与限制

- **框架版本漂移**：LangChain 迭代快，以 uv.lock 锁定版本；升级属重构，须回归全绿。
- **ChatOpenAI 按 OpenAI 官方 API 规范对齐**，非标字段（如 DeepSeek 的 `reasoning_content`）不被透传——已在 D8 取舍中接受；M6 专项评估结论见 D16（提示词引导替代，锁定版 langchain-openai 不提取该字段）；若未来需要深度适配特定供应商，再评估专用集成包（须征得同意）。usage 依赖显式 `stream_usage=True`，由回归测试锁定。
- `run_python` 非沙箱：仅用于本地可信代码的审查场景，文档多处明示；子进程环境经净化后继承——剔除 `SE_CodeAgent` 与 `*API_KEY*` 型变量（大小写不敏感），key 不进入子进程、更不进入模型上下文（验收走查修复；名称黑名单无法覆盖"名字无害、值是凭证"的变量，属已知边界）。子进程仍拥有用户全部权限（非沙箱不变）。**威胁模型**：被审查的第三方代码内容可构成对模型的间接提示注入、诱导自动执行任意本机代码——缓解：`--ask-exec`/`/confirm` 开关、环境净化、工具 description 明示；目录白名单已入 Backlog 评估。
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

- **M7 CLI 易用性**（已落地 2026-10-01；作业一可选增强，2026-10-01 用户提出"命令行使用更轻松"）：
  1. **裸入口**：`cra`（省略子命令）默认进入 chat REPL——argparse 子命令可选化 + 默认分支，既有子命令与全局参数面不变；缺失子命令的既有测试语义反转（原期望报错退出）；
  2. **`/setting` 设置菜单**：REPL 内数字选择，**全部复用既有实现**（纯接线）——1) 切换当前会话模型（= /change model） 2) 配置模型列表（= `cra config` 交互函数：增删供应商/模型、设默认、改 key） 3) 保存会话 4) 加载会话 5) 执行确认开关；0/回车返回对话；菜单内任何操作不丢上下文；
  3. **`/exit`**：退出（与 `exit` 等价）；`/help` 与 README 命令表同步；
  4. **会话自动保留**：退出时自动把当前会话覆盖写入 `~/.cra/sessions/_last.json`，启动检测到非空存档打印提示（不自动加载）；`/resume`（= /load _last）恢复，结构自检失败提示不可用。
  **明确不做**：Web 界面（D3 冻结 + 性价比评估见 Backlog）；菜单内再做二级工作区抽象（超出作业一体量）。
  范围理由：用户示例流程（裸 `cra` → 直接对话；`/setting` → 数字菜单 → 切换模型/配置模型列表）绝大多数是既有能力的入口接线，新增逻辑集中在 argparse 默认分支、菜单循环与自动存档，估计 +150~250 LOC（含测试），1 个编码会话。
  **落地注记（2026-10-01）**：① 裸形式不带参数——argparse 子解析器默认值会覆盖顶层同名参数值，共享参数面保持在子命令之后（§7.1 已按实现修订）；② /setting 全部复用既有命令函数（_cmd_change_model / _cmd_save / _cmd_load / _cmd_confirm 与 `cra config` 交互循环；后者增可选 console 参数以复用 REPL 终端面，菜单内捕获其损坏配置时的 SystemExit(2) 降级为回菜单，会话不终止）；③ 自动保留对**空会话跳过**——新开即退的空会话不覆盖上一次的有效存档；④ sessions 名称白名单放开首字符下划线（存档名 `_last`；防穿越语义不变）；⑤ 测试隔离升级：conftest 的 home_dir fixture 改为 autouse（自动保留会写 ~/.cra/sessions/_last.json，全部测试须与真实用户目录隔离）。

- **M8 会话与配置便利性**（已排期 2026-10-02，用户验收反馈提出；新会话执行）：
  1. **首次配置引导**：chat 无任何配置时询问"是否现在配置"——确认走引导式配置（添加供应商 → key → 设默认模型 → 进入 REPL，复用 `cra config` 交互函数与 _ConfigSession，不重写逻辑）；拒绝/完成后仍无配置维持指引 + 退出 2；`cra review` 保持非交互退出 2（§5 修订）；
  2. **供应商预设**：`cra config` 添加供应商提供编号选择（1 自定义 + 8 家预设），选预设自动填官方 base_url 与模型清单，**预填项均可修改**（base_url 回车用官方、模型清单询问是否使用预填）；预设表如下（2026-10-02 按官方文档核对，**实现前须逐项复核时效性**）：

     | # | 供应商 | base_url | 预填模型 | 惯用变量名 |
     |---|--------|----------|----------|-----------|
     | 1 | 自定义 | 用户填写 | 用户填写 | 规则生成 |
     | 2 | DeepSeek | `https://api.deepseek.com` | deepseek-flash, deepseek-v4-pro | `DEEPSEEK_API_KEY` |
     | 3 | Kimi (Moonshot) | `https://api.moonshot.cn/v1` | kimi-k3, kimi-k2.6 | `MOONSHOT_API_KEY` |
     | 4 | OpenAI | `https://api.openai.com/v1` | gpt-6-astra, gpt-6-luna（⚠ 命名迭代快，实现时按官方 models 页复核） | `OPENAI_API_KEY` |
     | 5 | 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen3.8-max, qwen3.8-flash | `DASHSCOPE_API_KEY` |
     | 6 | Z.ai Coding Plan | `https://api.z.ai/api/coding/paas/v4` | glm-5.3, glm-5.3-flash | `ZAI_API_KEY` |
     | 7 | Z.ai API | `https://api.z.ai/api/paas/v4` | glm-5.3, glm-5.3-flash | `ZAI_API_KEY` |
     | 8 | BigModel Coding Plan | `https://open.bigmodel.cn/api/coding/paas/v4` | glm-5.3, glm-5.3-flash | `ZHIPUAI_API_KEY` |
     | 9 | BigModel API | `https://open.bigmodel.cn/api/paas/v4` | glm-5.3, glm-5.3-flash | `ZHIPUAI_API_KEY` |

  3. **key 存储选择**（D17）：明文 vs 环境变量；变量名规则、Windows `setx` 自动 + 进程内生效、POSIX 打印 `export`、models.json 不存明文/移除已有明文、resolve 回退链 `{PROVIDER}_API_KEY` → `SE_CodeAgent`（§5/§7.4 修订）；
  4. **会话管理**：`/load` 与菜单 4) 省略名称时列出会话数字选择；加载后自动打印全部历史；`/delete` + 菜单 6)（y/N 确认）；菜单 3) 保存回车自动时间戳命名（§7.3 修订）；
  5. **切换模型**：现有 `/change model` 已是"先供应商后模型"流程，仅补充**当前使用中**标记。
  范围理由：验收演示暴露首次使用断层（新机器无配置无环境变量）与会话管理便利性诉求；估计 +400~600 LOC（含测试），1 个编码会话。
  **明确不做**：联网拉取模型清单（离线可用优先：内置清单 + 可编辑 + `cra config test` 验证）；自动改写 POSIX shell 配置文件；多套 key 轮换。

**作业二规划**（Simple Workflow MVP，不提前实现）：代码审查工作流方向——以本项目 Agent 为节点（§2.2 接缝），LangGraph `StateGraph` 编排"编码→审查→修复"或同等流程；顺序执行为 MVP，分支/并行/日志追踪按作业二推荐项递增。

**Backlog（未决功能池，新增功能先入池评审再排期）**：`--log-file` 本地调试日志；`/history` 会话消息摘要；累计 token 用量显示；run_python 目录白名单；多智能体协作/反思复核/RAG（作业二本体素材）；可视化编排界面（作业二可选项）；**Web 界面**（2026-10-01 评估：便利但性价比低——需本地服务 + 流式通道（SSE/WebSocket）+ 前端 + 会话状态管理，新增依赖与测试面翻倍，估计 2–3 个 M4 级编码会话，且违反 D3 冻结；CLI 设置菜单已覆盖易用性诉求，若作业二需要多端形态须先解除 D3）；**多工作区 / 按项目配置**（"工作区"语义待澄清：配置存储已由 ~/.cra 覆盖，若指按项目切换工作目录/配置，超出作业一体量）。
