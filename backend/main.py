"""FastAPI 服务入口：REST + SSE.

启动方式（仓库根目录）：
    uvicorn backend.main:app --port 8000

会话图状态由 AsyncSqliteSaver 持久化到 backend/data/checkpoints.sqlite，
业务数据（会话/消息/报告）存 backend/data/deep_research.db。
"""

import json
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

# graph_def 导入副作用：加载 .env 并把 research/src 注入导入路径，须先于其他导入
from backend import db
from backend.graph_def import build_graph
from backend.runner import SessionManager

CHECKPOINT_PATH = db.DATA_DIR / "checkpoints.sqlite"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    saver_ctx = AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_PATH))
    saver = await saver_ctx.__aenter__()
    manager = SessionManager(build_graph(saver))
    manager.recover_on_startup()
    app.state.manager = manager
    yield
    await saver_ctx.__aexit__(None, None, None)


app = FastAPI(title="Deep Research Server", lifespan=lifespan)

# 前端开发服务器（Vite 默认 5173）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 请求体 =====


class CreateSessionBody(BaseModel):
    question: str


class ResumeBody(BaseModel):
    feedback: str


def _get_manager(app: FastAPI) -> SessionManager:
    manager = getattr(app.state, "manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="服务尚未完成初始化")
    return manager


def _require_session(session_id: str) -> dict:
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


# ===== 会话 =====


@app.post("/api/sessions", status_code=201)
async def create_session(body: CreateSessionBody):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question 不能为空")
    session_id = _get_manager(app).start_session(question)
    return {"id": session_id, **db.get_session(session_id)}


@app.get("/api/sessions")
def list_sessions():
    return db.list_sessions()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    return _require_session(session_id)


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str):
    _require_session(session_id)
    return db.list_messages(session_id)


@app.post("/api/sessions/{session_id}/resume")
async def resume_session(session_id: str, body: ResumeBody):
    session = _require_session(session_id)
    if session["status"] != "waiting_review":
        raise HTTPException(status_code=409, detail=f"会话当前状态为 {session['status']}，无需人工输入")
    _get_manager(app).resume(session_id, body.feedback.strip())
    return {"ok": True}


@app.get("/api/sessions/{session_id}/events")
async def session_events(session_id: str):
    """SSE：实时推送该会话的执行事件（node_update / message / interrupt / report_done / error）."""
    _require_session(session_id)
    manager = _get_manager(app)
    queue = manager.subscribe(session_id)

    async def event_stream():
        try:
            while True:
                event = await queue.get()
                data = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['event']}\ndata: {data}\n\n"
        finally:
            manager.unsubscribe(session_id, queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ===== 报告 =====


@app.get("/api/reports")
def list_reports():
    return db.list_reports()


@app.get("/api/reports/{report_id}")
def get_report(report_id: str):
    report = db.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    return report


# ===== 统计 =====


@app.get("/api/stats/active")
def stats_active():
    return {"active": db.count_active_sessions()}
