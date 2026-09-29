# Design.md — 代码审查助手 code-review-agent

《软件工程与项目实践》第 1 次作业 · 提交截止 2026-10-07 24:00
平台环境：Windows + Git Bash（PowerShell/CMD 兼容）。

## 1. 项目概述

`cra` 是一个基于 LLM 的命令行代码审查助手：给定一段/一个/一组代码文件（任意文本型源码、脚本与配置文件，Python 为主），它以代码审查员身份给出问题清单（分级、定位、修改建议），并能在多轮对话中自主调用工具（读文件、运行代码）核实问题。

- 交付物：GitHub 私有仓库打包 + README.md / Design.md +（可选，用户自理）1 分钟演示视频；备份提交"学号姓名.rar"（<200M）至课程"001Homework1"。
- 评分权重：功能完整性 40%（核心可用、边界情况）· Agent 架构 30%（设计模式、架构清晰度）· 代码质量 20% · 文档 10%。

## 2. 范围

| 级别 | 内容 |
|------|------|
| Must | 审查代码（任意文本型源码，Python 为主；M2 验收意见放开，原为仅 Python）；文件读取工具（列目录 + 读单文件）；代码执行工具（subprocess 限时 10s，本机运行非沙箱，限制明示）；多轮上下文记忆；指数退避重试；CLI（彩色流式输出，入口 `cra`）；pytest 测试集 |
| Should | 审查结果导出 Markdown 报告；批量审查多文件 |
| Won't | Web 界面；多语言**执行**（run_python 仅运行 Python，任意语言源码的审查不受限——M2 验收意见修订，原为"多语言支持（非 Python 文件给友好提示即止）"）；Docker 沙箱；联网搜索工具 |

## 3. 总体架构

```
┌──────────┐   user input    ┌──────────────┐   tools (JSON)  ┌──────────────┐
│  cli.py  │ ──────────────► │   agent.py   │ ──────────────► │ DeepSeek API │
│  REPL /  │ ◄────────────── │  ReAct loop  │ ◄────────────── │ (openai SDK) │
│  render  │  stream + color └──────┬───────┘  tool_calls     └──────────────┘
└──────────┘                        │
                          ┌─────────▼────────┐
                          │ tools / registry │
                          └──────────────────┘
                            list_dir / read_file / run_python
```

模块划分（src 布局，包目录固定为 `src/cra`）：

```
src/cra/
  cli.py        # argparse + REPL 循环，rich 彩色流式渲染
  agent.py      # ReAct 循环：messages 维护、tool_calls 分发、最大轮次、上下文截断
  llm.py        # DeepSeek 客户端封装（openai SDK、指数退避重试）
  tools/
    __init__.py # 工具注册表：名称 → (JSON schema, handler)
    fs.py       # list_dir、read_file
    exec_py.py  # run_python（subprocess 限时执行）
  config.py     # 环境变量读取与默认值（见 §6 配置表）
  prompts.py    # 系统提示词与审查输出格式约定
scripts/
  smoke_tools.py   # M0 冒烟测试（入库留存）：用最小哑工具（如 echo）验证 function calling 通路，与业务工具 schema 无关
tests/
  fixtures/buggy_samples/   # 预埋已知问题的样例（黄金回归用例）
```

### 3.1 Agent 循环（核心，评分权重最高）

"输入 → 推理 → 工具调用 → 输出"的实现：

1. 用户输入追加进 `messages`；系统提示词常驻首位（角色 + 输出格式 + 工具使用策略）。
2. 调用 LLM，传入工具注册表的 JSON schema（function calling）。**工具轮次一律非流式调用**（需要完整 `tool_calls`，`usage` 直接可取）；仅最终回答使用流式。
3. 若响应含 `tool_calls`：逐个执行 handler → 结果作为 tool 角色消息追加 → 回到 2；**最大工具轮次 10**，超过则强制要求模型给出结论，防止失控。
4. 否则流式输出最终回答：rich Console 逐段打印 + 终块 Markdown 渲染，**不使用 rich.Live**（Windows 传统控制台下 Live 重绘不可靠）。
5. **上下文记忆** = `messages` 列表在 REPL 会话内持续存在。截断：以最近一次响应的 `usage.total_tokens` 超过 `CRA_MAX_CONTEXT_TOKENS`（默认 100000）为阈值（流式模式下需 `stream_options={"include_usage": true}` 才能在末块拿到 usage）；以**完整轮次**（用户消息 → 最终回答，含其间全部 tool 消息）为原子单位从最旧丢弃，保证 `tool_calls` 与其配对的 `tool` 消息不被拆散（拆散会导致 API 400）；系统提示词永远保留。

### 3.2 工具 schema

| 工具 | 参数 | 行为与限制 |
|------|------|-----------|
| `list_dir` | `path` | 返回两层目录树；忽略 `.venv`、`__pycache__`、`.git`、`.pytest_cache`、`.ruff_cache` 与点开头的隐藏目录 |
| `read_file` | `path`, `start?`, `end?` | 带行号返回内容；单次上限 **400 行**，超出提示用 start/end 分段；`start`/`end` 为 1-based 闭区间，非法区间（start>end、负数）返回结构化错误；二进制/解码失败返回结构化错误信息（回传模型） |
| `run_python` | `code` 或 `path`（二选一）, `timeout?` | subprocess 运行，工作目录=进程启动目录（见下方通用规则）；默认 10s（`CRA_EXEC_TIMEOUT` 可调），Windows 下用 `taskkill /T /F` 强杀整棵进程树；返回 stdout/stderr/退出码；stdout/stderr 各截断至 **2000 字符**；**本机直接运行，非沙箱** |

通用规则：三工具的相对路径一律相对**进程启动目录（cwd）**解析，不做"项目根"推断，run_python 的子进程工作目录同此；所有文件读取与子进程输出捕获**显式 `encoding='utf-8', errors='replace'`**（Windows 默认 cp936，否则 UTF-8 源码会读成乱码、子进程中文输出会解码崩溃），子进程环境可另设 `PYTHONIOENCODING=utf-8`。

### 3.3 系统提示词要点

角色为资深代码审查员（Python 为主深耕，任意语言的源码、脚本与配置文件均可审查；执行验证仅 Python，其他语言静态审查并标注"不确定"）；输出固定结构：【严重/一般/建议】分级 + `文件:行号` + 问题 + 修改建议（修改建议独立成行、紧跟问题描述）；先调用工具核实再下结论；不确定的问题明确标注"不确定"，禁止编造行号或代码。该输出格式为**提示词约定**，全系统不做程序化校验/解析（不写格式断言测试）。

### 3.4 代码设计规范落点（第3讲：规范 = 风格 + 设计）

风格层由 ruff 强制（§6）；设计层按以下原则执行，M3 Checklist 核对：

| 原则 | 本项目落点 |
|------|-----------|
| 单一职责 | cli 只管交互与渲染；agent 只管循环与状态；llm 只管 API 与重试；tools 只管工具执行；config 是环境变量唯一出入口 |
| 依赖与模块化 | 依赖方向单向：cli → agent → {llm, tools}；tools/llm 不反向 import agent/cli，不越层调用（cli 不直接调 llm）。**M0 过渡期豁免**：agent.py 尚未存在时允许 cli 直连 llm，M1 起必须收敛（列入 M1 重构点） |
| 数据封装 | messages 结构与截断规则只存在于 agent.py；工具返回统一为字符串（错误也是字符串），注册表屏蔽 handler 细节 |
| 异常处理 | 按 §5：异常要么被结构化回传模型，要么给出用户可读报错，不允许静默吞掉 |
| 性能 | 流式输出保证感知性能；工具调用有超时；上下文有截断阈值 |
| 安全 | key 只在内存流转、永不落盘；run_python 限时 + 截断 + 限制声明（第3讲 Gab 教训：规范一丢，漏洞就来） |

## 4. 关键设计决策及理由

| 决策 | 理由 |
|------|------|
| 手写 ReAct 循环，不用 LangChain 等框架 | 架构分占 30%：核心循环 <200 行，可控、可讲、可测；框架黑盒不利于架构叙述 |
| DeepSeek + openai SDK（OpenAI 兼容） | 国内直连、成本低；换服务商只改 `CRA_BASE_URL` 与 `CRA_MODEL`。注意 DeepSeek 模型线已更新（`deepseek-flash` / `deepseek-v4-pro`），M0 必做 function calling 冒烟测试 |
| function calling 降级预案 | 若 M0 冒烟发现 tool_calls 行为异常：降级为"JSON 模式"——提示词要求模型输出 `{"tool": ..., "args": ...}` 由循环解析，接口层按可替换设计 |
| run_python 用 subprocess + 限时，不做沙箱 | 作业范围权衡；风险用"限制明示 + 超时强杀 + 输出截断"控制，README 与 `cra chat --help` 均声明（Checklist 有核对项） |
| 上下文记忆 = messages 列表 + 轮次原子截断 | 原生、可解释、易测；摘要属 Should 外功能，明确不做 |
| 仅 CLI | 作业允许二选一；聚焦 Agent 核心能力，Web 界面为 Won't |
| 里程碑即增量（SCRUM 式迭代增量，第3讲） | 每个里程碑结束都交付一个"潜在可交付"的可用增量：M0 可单轮对话 → M1 可多轮工具审查 → M2 边界健壮 → M3 收尾；用户验收即持续反馈，回归测试保护既有增量；也为结对项目在本项目上"增量开发 + 重构"铺路 |
| 为结对项目预留扩展接缝 | 工具注册表即插即用（新工具 = 注册一项）；CLI 子命令扩展位（Should 的 batch/report）；prompts 独立成模块——结对项目增量开发时无需重构骨架 |

## 5. 错误处理与重试

- 启动校验：`SE_CodeAgent` 非空，否则打印设置指引后退出；**不校验格式**（合法格式的无效 key 由 API 认证环节暴露）。
- LLM 调用：超时/限流/连接错误 → 指数退避重试（1s/2s/4s，最多 3 次；**请求级超时 60s**，llm.py 常量，不用 SDK 默认 600s），仍失败给出友好报错与排查建议；**认证类错误（401/403）快速失败、不重试**，输出可读的 key 排查指引。
- 工具执行：handler 内所有异常捕获 → 结构化错误文本作为 tool 结果回传模型（模型可自行调整策略），REPL 不崩溃。模型返回**未注册工具名或参数不符 schema** 时同样以结构化错误文本回传（手工轻校验，不为此引入 jsonschema 依赖），不触发 API 重试。
- `run_python`：超时强杀进程树（Windows 下确保子进程一并终止）；stdout/stderr 截断防刷屏。
- `Ctrl+C`：中断当前生成——**任何中断（含工具执行中）必须把 messages 回滚到本轮开始前的合法前缀，丢弃未完成的 assistant/tool 消息并提示该轮作废**（残留半截消息会导致下一轮 API 400）；输入 `exit` 或再次 `Ctrl+C` 退出 REPL（Windows 控制台无 Ctrl+D EOF 语义）。

## 6. 工程规格（编码会话照此施工）

- 包管理：uv（≥0.12），`requires-python = ">=3.13"`，`.python-version` 固定 `3.13`。
- **uv init 陷阱（M0 必读）**：仓库目录非空（已有 README.md 与 docs/，**不得覆盖或删除**）；`uv init --package --name code-review-agent` 生成的包目录名是 `code_review_agent`，必须重建/重命名为 `src/cra`，并在 pyproject 同步 `[tool.hatch.build.targets.wheel] packages = ["src/cra"]`。
- 运行依赖：`openai`、`rich`。开发依赖：`pytest`、`ruff`（`uv add --dev`）。
- 入口与 CLI 面：`[project.scripts]` → `cra = "cra.cli:main"`；子命令 `cra chat` 进入 REPL（本作业范围）；保留子命令扩展位（Should 的 batch / report 未来挂载）。REPL 内置命令（M1 实现）：`exit` 退出；`/clear` 将 messages **重置为初始状态（仅含系统提示词）**，角色与格式契约保留。空输入/纯空白直接进入下一轮提示、不调 API；多行粘贴不做特殊支持，系统提示词引导"整段代码先落盘成文件再审查"。
- 配置（环境变量，统一在 `config.py` 读取）：

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `SE_CodeAgent` | 是 | — | DeepSeek API key（仅启动校验非空） |
| `CRA_MODEL` | 否 | `deepseek-flash` | 模型名 |
| `CRA_BASE_URL` | 否 | `https://api.deepseek.com` | OpenAI 兼容接入点 |
| `CRA_EXEC_TIMEOUT` | 否 | `10` | run_python 超时（秒） |
| `CRA_MAX_CONTEXT_TOKENS` | 否 | `100000` | messages 截断阈值（tokens） |

- ruff：默认规则 + `line-length = 100`；pytest `testpaths = ["tests"]`，**测试内禁止真实网络调用（LLM 一律 mock），真实网络仅限 scripts/smoke_tools.py**。
- LLM 请求参数：`temperature=0.2`（输出稳定，黄金用例可复现）；`max_tokens` 沿用服务商默认。
- `.gitignore`：`.venv/`、`__pycache__/`、`.pytest_cache/`、`.ruff_cache/`、`dist/`、`.env`、`*.pyc`。
- 仓库初始化（M0 执行）：`git init` → 首个 commit → `gh repo create code-review-agent --private --source .`（gh 已登录，若失败则提示用户网页手动建）。

## 7. AI 辅助开发披露

本项目采用"人机分工"的工程实践：**人**负责需求决策、架构设计评审、各里程碑验收（运行 + 抽查 + 测试）、推送与提交把关；**AI 编码会话**按本文档的冻结决策与工程规格生成代码与测试。流程为：设计文档先行（本文档经人工评审后才动工）→ 每里程碑由独立会话按 DoD 实现 → 人工验收通过后推送。过程中的估算、Checklist、复盘记录见第 8 节与 `docs/PROGRESS.md`。

## 8. PSP 附录（轻量诚实版）

**开发过程定位（第3讲）**：本计划即**增量开发**——每个里程碑交付一个可用增量，用户验收即持续反馈，回归测试保护既有增量，验收意见回灌下一里程碑（迭代）。代码审查按三级落地：**静态审查**（ruff + pytest 自动执行）、**动态审查**（用户验收中的实际运行检查）、**代码走查**（用户里程碑验收 + 会话收尾总结，审查范围 = 该里程碑 diff）、**结果记录**（PROGRESS.md 验收记录与一句意见）；M3 的 Checklist 是走查的最后一环。

### 8.1 估算表（开发前估算，收尾时对照实际填偏差）

| 项 | 估算 | 实际（收尾填） |
|----|------|----------------|
| 总规模 | ~1000 LOC（含测试，git 统计为准） | |
| 人工投入 | 设计评审 0.5h + 4×里程碑验收 0.25h + 收尾 0.5h = **2.0h** | |
| AI 生成 | 全部实现代码与测试初稿 | |
| 日历时间 | 2–3 天（4 个编码会话，每个 30–90 分钟） | |

### 8.2 代码评审 Checklist（收尾门禁逐项执行，结果写回此处）

- [ ] 全仓库 grep 无明文密钥；`SE_CodeAgent` 只经 `config.py` 从环境变量读取
- [ ] 无裸 `except:`；所有工具异常均回传模型而非静默吞掉
- [ ] `run_python` 有超时强杀与输出截断
- [ ] Agent 循环有最大工具轮次（10）限制
- [ ] 消息截断按完整轮次原子丢弃，`tool_calls`/`tool` 配对不被拆散，系统提示词保留
- [ ] `cra chat --help` 与 README 含"非沙箱、限时执行"限制声明
- [ ] README 每条命令逐条实测可复制执行
- [ ] pytest 覆盖：工具 handler、消息组装、截断逻辑、CLI 参数解析
- [ ] ruff 全绿；公开函数有类型标注；命名自解释、无无意义缩写
- [ ] 模块依赖方向符合 §3.4（无反向 / 越层 import）
- [ ] `git status` 干净：无 venv/缓存/密钥文件被跟踪
- [ ] Design.md 与实现一致（模型名、工具名、模块名、入口、配置项）

### 8.3 复盘、缺陷记录与 PIP（收尾时填写）

- 缺陷记录：各里程碑验收中发现的问题数与去向（现场修复 / 遗留说明）
- 估算偏差：（人工/规模/日历各实际 vs 估算）
- 过程问题：
- PIP（下一项目——结对项目将在本项目上增量开发与重构——的改进项）：

## 9. 风险与限制

- DeepSeek V4 系列的 function calling 行为以 M0 冒烟测试为准，已预留 JSON 模式降级预案（见 §4）。
- `run_python` 非沙箱：仅用于本地可信代码的审查场景，文档多处明示；子进程默认**继承父环境变量（含 `SE_CodeAgent`）**，属非沙箱已知边界，如需收紧可传入净化后的 env。
- 审查质量依赖模型能力：对专有框架/最新库版本可能误报，提示词已要求标注"不确定"。
- 上下文超长仅做轮次截断不做摘要（超出 Must 范围）。
