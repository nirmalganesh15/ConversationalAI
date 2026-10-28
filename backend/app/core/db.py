"""Tiny SQLite repository for users and chat sessions.

SQLite keeps the demo single-container; the repository functions are the seam where
Postgres would slot in for production (see docs/productionization.md).
"""

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from app.core.security.auth import Principal, hash_password, verify_password
from app.core.security.policy import Role

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    agent TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sessions_user ON sessions(user_id, updated_at DESC);
"""


class ChatSession(BaseModel):
    id: str
    agent: str
    title: str
    created_at: str
    updated_at: str


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- users -------------------------------------------------------------
    def upsert_user(self, username: str, password: str, role: Role) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO users (id, username, password_hash, role) VALUES (?, ?, ?, ?)
                   ON CONFLICT(username) DO UPDATE SET password_hash=excluded.password_hash,
                   role=excluded.role""",
                (str(uuid.uuid4()), username, hash_password(password), role.value),
            )

    def authenticate(self, username: str, password: str) -> Principal | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return Principal(user_id=row["id"], username=row["username"], role=Role(row["role"]))

    # --- sessions ------------------------------------------------------------
    def create_session(self, user_id: str, agent: str, title: str) -> ChatSession:
        now = datetime.now(UTC).isoformat()
        session = ChatSession(
            id=uuid.uuid4().hex, agent=agent, title=title[:80], created_at=now, updated_at=now
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                (session.id, user_id, agent, session.title, now, now),
            )
        return session

    def get_session(self, user_id: str, session_id: str) -> ChatSession | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
            ).fetchone()
        return ChatSession(**{k: row[k] for k in ChatSession.model_fields}) if row else None

    def list_sessions(self, user_id: str) -> list[ChatSession]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100",
                (user_id,),
            ).fetchall()
        return [ChatSession(**{k: r[k] for k in ChatSession.model_fields}) for r in rows]

    def touch_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), session_id),
            )

    def delete_session(self, user_id: str, session_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
            )
        return cur.rowcount > 0
