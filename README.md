# Deep Research System

基于 LangGraph 的多智能体深度研究系统，配有完整的 Web 前后端。输入一个研究方向，系统会先与你协商生成研究简报（人工审核/修订），确认后派出多个研究员并行检索，最终汇总生成结构化研究报告。

> 原始教程项目（5 个渐进式 notebook）位于 [`research/`](./research)，其文档见 [research/README.md](./research/README.md)。

## 功能特性

- **简报协商循环**：方向模糊时模型主动反问澄清，简报生成后进入人工审核，可无限轮修订（含轮数软警告）
- **多智能体并行研究**：supervisor 协调多个 researcher 子代理，并行调用 Tavily 检索 + think_tool 反思，单研究员搜索次数硬上限
- **人工审核中断**：全流程唯一的 interrupt 入口，支持服务重启后从断点恢复
- **Web 多窗口**：每个浏览器标签一个独立研究会话，互不阻塞，可同时进行多项研究
- **实时流式推送**：SSE 推送节点进度、聊天消息、审核请求与报告完成事件
- **研究报告中心**：所有已完成报告落库，附协商定稿的简报，Markdown 渲染阅读

## 目录结构

```
deep_research_from_scratch/
├── frontend/               # 前端（Vite + React + TS + Tailwind v4 + Zustand）
│   └── src/
│       ├── pages/          # 聊天页 / 报告列表 / 报告阅读
│       ├── components/     # 侧边栏（任务计数、会话列表、报告入口）
│       ├── api.ts          # REST + SSE 客户端
│       └── store.ts        # 全局状态
├── backend/                # 后端（FastAPI + LangGraph）
│   ├── graph_def.py        # 深度研究图的构建模块（抽取自 research/notebooks/5_full_agent copy.py）
│   ├── runner.py           # 会话执行器：每会话一个 asyncio 任务，astream → interrupt → resume 循环
│   ├── db.py               # SQLite 业务库（会话 / 消息 / 报告）
│   ├── main.py             # FastAPI 入口：REST + SSE
│   └── data/               # 运行时数据（gitignore）：deep_research.db + checkpoints.sqlite
├── research/               # 原教程项目（源头）
│   ├── notebooks/          # 5 个教程 notebook（唯一修改入口）+ 自包含运行脚本
│   ├── src/                # 由 notebook %%writefile 生成的源码包（勿直接改）
│   └── pyproject.toml      # 包定义
├── .env                    # API keys（根目录）
└── AGENTS.md               # 开发规范
```

## 环境准备

**Python 环境**：必须使用项目自带的 `.venv`（Python 3.12），系统默认 Python 3.14 与 `research/pyproject.toml` 的版本约束（<3.14）不兼容。

```powershell
# 在仓库根目录执行（已安装过则跳过）
.venv\Scripts\python.exe -m ensurepip
.venv\Scripts\python.exe -m pip install -e ./research          # 教程源码包（editable）
.venv\Scripts\python.exe -m pip install fastapi "langgraph-checkpoint-sqlite"
```

**API Keys**：在根目录 `.env` 中配置（`research/src` 的模块在 import 时即读取）：

```
ZHIPUAI_API_KEY=...     # 智谱 GLM（当前所有模型节点使用 openai:glm-5.3-flashx）
TAVILY_API_KEY=...      # 联网检索
```

> `.env` 已被 `.gitignore` 排除，不会随仓库上传；首次使用请参照上方说明自行创建。

**前端依赖**：

```powershell
cd frontend; npm install
```

## 启动

```powershell
# 终端 1：后端（仓库根目录，端口 8000）
.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000

# 终端 2：前端（端口 5173，/api 已代理到 8000）
cd frontend; npm run dev
```

打开 http://localhost:5173 即可使用。

## 使用流程

1. **新建研究**：侧边栏「+ 新研究」→ 描述研究方向（Ctrl+Enter 提交）
2. **协商简报**：模型可能先反问澄清问题；研究简报生成后会作为助手消息出现在对话流中
3. **审阅**：对话流末尾出现人工审核卡片，回复 `OK` 确认进入研究阶段（该回复保留在对话记录中），或直接输入修改意见让模型修订简报（修订版简报会以新消息追加，可多轮）
4. **并行研究**：supervisor 派出多个研究员并行检索，界面实时显示进度
5. **查看报告**：完成后顶部出现「查看研究报告」入口，报告中心（侧边栏左下角）可随时回看

侧边栏左上角的数字 = 进行中的研究任务数（研究中 + 待审阅），每 5 秒自动刷新。

## 后端 API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/sessions` | 创建研究会话（body: `{question}`），立即启动图执行 |
| GET | `/api/sessions` | 会话列表（标题、状态、时间） |
| GET | `/api/sessions/{id}` | 会话详情 |
| GET | `/api/sessions/{id}/messages` | 完整聊天记录 |
| POST | `/api/sessions/{id}/resume` | 提交人工审核反馈（`OK` 或修改意见） |
| GET | `/api/sessions/{id}/events` | **SSE** 实时事件流 |
| GET | `/api/reports` | 报告列表 |
| GET | `/api/reports/{id}` | 报告详情（简报 + 全文 Markdown） |
| GET | `/api/stats/active` | 进行中任务数 |

**SSE 事件类型**：`status`（连接建立时补发的状态快照）、`node_update`（节点进度）、`message`（聊天消息）、`interrupt`（等待人工审核，附当前简报）、`report_done`（报告完成）、`error`。

## 架构要点

- **会话 = LangGraph thread_id**：每个研究会话对应图中一个独立线程；图执行跑在 FastAPI 的 asyncio 后台任务里，不阻塞请求
- **interrupt / resume 协议**：`review_brief` 节点 `interrupt()` 暂停 → 后端置会话为 `waiting_review` 并广播 SSE → 前端提交反馈 → `Command(resume=feedback)` 续跑
- **双库持久化**：图状态存 `AsyncSqliteSaver`（checkpoints.sqlite），业务数据存独立 SQLite（deep_research.db），避免锁竞争；服务重启后 `waiting_review` 会话仍可续跑，`running` 会话标记为失败
- **模型配置**：在 [backend/graph_def.py](backend/graph_def.py) 顶部统一调整（当前全部智谱 `openai:glm-5.3-flashx`，经 OpenAI 兼容端点接入）；研究预算参数（`MAX_RESEARCHER_ITERATIONS`、`MAX_RESEARCHER_SEARCHES`、`MAX_CONCURRENT_RESEARCHERS`、`MAX_BRIEF_REVIEW_ROUNDS`、`BRIEF_ONLY_DEBUG_MODE`）已支持在 `.env` 中配置，未设置时使用代码默认值。修改 `.env` 后需重启后端生效
- **教程项目**：`research/notebooks/` 是唯一修改入口，`research/src/` 由 notebook 的 `%%writefile` 生成，勿直接编辑

## 常见问题

- **启动报 API Key 缺失**：`.env` 必须在仓库根目录，且后端从根目录启动（`backend/graph_def.py` 按相对路径加载）
- **前端 401/CORS**：开发模式下 Vite 已将 `/api` 代理到 8000，无需额外配置；若改端口请同步 [frontend/vite.config.ts](frontend/vite.config.ts)
- **服务重启后会话丢失**：`running` 状态的会话会被标记为失败（图执行不可跨进程迁移）；`waiting_review` 的会话不受影响，重新提交反馈即可继续

## 致谢

本项目基于 LangChain 官方开源教程项目 [langchain-ai/deep_research_from_scratch](https://github.com/langchain-ai/deep_research_from_scratch) 构建——`research/` 目录完整保留了其 5 个渐进式教程 notebook（scope → research agent → MCP → supervisor → full agent），它们是本项目全部核心架构的源头。感谢 LangChain 团队与社区贡献者的高质量教程。在此之上，本项目增加了 FastAPI 后端、React 前端、SQLite 持久化与人工审核回环等工程化封装。
