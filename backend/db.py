"""SQLite 业务库：会话 / 消息 / 报告 三张表.

与 LangGraph 的 checkpointer 库分开存放，避免锁竞争。
单文件低并发场景，直接使用标准库 sqlite3，每次操作独立连接。
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from typing_extensions import Literal

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "deep_research.db"

SessionStatus = Literal["running", "waiting_review", "completed", "failed"]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_session (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'running',
                brief      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_message (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES research_session(id),
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_report (
                id           TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL REFERENCES research_session(id),
                title        TEXT NOT NULL,
                brief        TEXT NOT NULL DEFAULT '',
                content      TEXT NOT NULL,
                generated_at TEXT NOT NULL
            );
            """
        )
        # 旧库迁移：research_session 增加 brief 列，并从报告表回填历史会话的定稿简报
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(research_session)")}
        if "brief" not in cols:
            conn.execute("ALTER TABLE research_session ADD COLUMN brief TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                UPDATE research_session SET brief = COALESCE((
                    SELECT brief FROM research_report WHERE research_report.session_id = research_session.id
                ), '') WHERE brief = ''
                """
            )


def _touch(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "UPDATE research_session SET updated_at = ? WHERE id = ?",
        (_now(), session_id),
    )


# ===== 会话 =====


def create_session(title: str) -> str:
    session_id = uuid.uuid4().hex[:12]
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO research_session (id, title, status, created_at, updated_at) VALUES (?, ?, 'running', ?, ?)",
            (session_id, title, now, now),
        )
    return session_id


def list_sessions() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, status, created_at, updated_at FROM research_session ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, status, brief, created_at, updated_at FROM research_session WHERE id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def update_session_brief(session_id: str, brief: str) -> None:
    """记录最新版研究简报：每次修订覆盖，简报确认后即停在此值（= 最终定稿）."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE research_session SET brief = ?, updated_at = ? WHERE id = ?",
            (brief, _now(), session_id),
        )


def set_session_status(session_id: str, status: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE research_session SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), session_id),
        )


def count_active_sessions() -> int:
    """进行中任务数 = running + waiting_review."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM research_session WHERE status IN ('running', 'waiting_review')"
        ).fetchone()
    return row["n"]


# ===== 消息 =====


def add_message(session_id: str, role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO research_message (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        _touch(conn, session_id)


def list_messages(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM research_message WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ===== 报告 =====


def save_report(session_id: str, title: str, brief: str, content: str) -> str:
    report_id = uuid.uuid4().hex[:12]
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO research_report (id, session_id, title, brief, content, generated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (report_id, session_id, title, brief, content, _now()),
        )
        _touch(conn, session_id)
    return report_id


def list_reports() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, session_id, title, generated_at FROM research_report ORDER BY generated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_report(report_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, session_id, title, brief, content, generated_at FROM research_report WHERE id = ?",
            (report_id,),
        ).fetchone()
    return dict(row) if row else None
