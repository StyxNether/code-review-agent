# code-review-agent · 代码审查助手

> ✅ 交付版（《软件工程与项目实践》第 1 次作业）。架构、接口规格与设计决策详见 [docs/Design.md](docs/Design.md)。

基于 LLM 的命令行代码审查工具。给它一段/一个/一组代码文件（任意文本型源码与配置文件，Python 为主），它以代码审查员身份输出分级问题清单，并能在多轮对话中自主调用工具（读文件、运行代码）核实问题。

《软件工程与项目实践》第 1 次作业项目。

## 功能

- 多轮对话式代码审查，具备上下文记忆（`/clear` 可随时清空重来）
- Agent 自主调用工具：列目录 / 读文件 / 运行 Python 代码核实问题（ReAct 循环 + function calling）
- 结构化审查结论：【严重/一般/建议】+ `文件:行号` + 修改建议
- 流式彩色输出；LLM 调用失败自动指数退避重试

## 架构

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

手写 ReAct Agent 循环（未使用 Agent 框架），设计理由与详细设计见 [docs/Design.md](docs/Design.md)。

## 快速开始

前置：Python 3.13+、[uv](https://docs.astral.sh/uv/)、DeepSeek API key。

```bash
# 1. 设置 API key（Windows；setx 后需新开终端生效）
setx SE_CodeAgent "sk-你的key"

# 2. 安装依赖
uv sync

# 3. 启动
uv run cra chat
```

## 使用示例

```text
$ uv run cra chat
code-review-agent 已就绪（模型：deepseek-flash）。提出审查需求即可；exit 退出，/clear 清空上下文。
你> 审查 src/cra/agent.py
⚙ read_file({"path": "src/cra/agent.py"})
  ↳ 完成（9xxx 字符）

【严重】src/cra/agent.py:42 — 工具调用缺少超时控制，可能永久阻塞。
修改建议：为 subprocess 调用设置 timeout 参数并捕获 TimeoutExpired。
【一般】src/cra/agent.py:57 — 异常被静默吞掉。
修改建议：将异常信息结构化回传模型。

你> 第 1 条为什么严重？
（多轮上下文延续，Agent 基于前文作答）
你> exit
再见。
```

## 配置

| 环境变量 | 必填 | 默认 | 说明 |
|----------|------|------|------|
| `SE_CodeAgent` | 是 | — | DeepSeek API key（仅启动校验非空） |
| `CRA_MODEL` | 否 | `deepseek-flash` | 模型名（OpenAI 兼容接口均可） |
| `CRA_BASE_URL` | 否 | `https://api.deepseek.com` | OpenAI 兼容接入点 |
| `CRA_EXEC_TIMEOUT` | 否 | `10` | 代码执行超时（秒） |
| `CRA_MAX_CONTEXT_TOKENS` | 否 | `100000` | 上下文截断阈值（tokens） |

## Agent 可用的工具

| 工具 | 说明 | 限制 |
|------|------|------|
| `list_dir` | 查看目录结构（两层，忽略缓存目录） | — |
| `read_file` | 带行号读取文件内容 | 单次 400 行，可分段 |
| `run_python` | 运行 Python 代码/文件核实问题 | 本机直接运行、非沙箱、默认 10 秒超时 |

## 限制

- `run_python` 在本机直接执行代码（**非沙箱、限时执行**：默认 10 秒，超时强制终止整棵进程树）：请勿在审查不可信代码时滥用执行功能；子进程会继承包括 API key 在内的环境变量。该限制同时明示于 `cra chat --help` 与工具 description。
- 审查支持任意文本型源码与配置文件；执行验证仅支持 Python（`run_python`），其他语言的结论以静态审查为准。
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

- [ ] 审查结果导出 Markdown 报告
- [ ] 批量审查多文件

---

本项目由 AI 编码会话辅助开发；人机分工与开发过程记录见 [docs/Design.md](docs/Design.md) §8。
