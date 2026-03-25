"""SQLite persistence for conversation history, user overrides, and feedback.

Tool-call observability is handled by LangSmith -- this module focuses
purely on conversation context and user actions that the app needs
across Streamlit reruns.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "store.db"


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create tables if they don't already exist."""
    conn = _connect(db_path)
    conn.executescript(
        """\
        CREATE TABLE IF NOT EXISTS conversation_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL DEFAULT (datetime('now')),
            role      TEXT    NOT NULL,
            content   TEXT    NOT NULL,
            metadata  TEXT
        );

        CREATE TABLE IF NOT EXISTS user_overrides (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL DEFAULT (datetime('now')),
            message_id   TEXT,
            machine_id   TEXT,
            date         TEXT,
            hour_window  TEXT,
            product      TEXT    NOT NULL,
            original_rec INTEGER NOT NULL,
            adjusted_rec INTEGER NOT NULL,
            reason       TEXT
        );

        CREATE TABLE IF NOT EXISTS user_feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL DEFAULT (datetime('now')),
            message_id TEXT    NOT NULL,
            rating     INTEGER NOT NULL,
            comment    TEXT
        );
        """
    )
    conn.commit()
    conn.close()


# ── Conversation history ────────────────────────────────────────────

def save_message(
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Persist a single chat message. Returns the row id."""
    conn = _connect(db_path)
    cur = conn.execute(
        "INSERT INTO conversation_history (role, content, metadata) VALUES (?, ?, ?)",
        (role, content, json.dumps(metadata) if metadata else None),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_conversation_history(
    *, db_path: str | Path = DEFAULT_DB_PATH
) -> list[dict[str, Any]]:
    """Return all messages ordered chronologically."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT id, timestamp, role, content, metadata "
        "FROM conversation_history ORDER BY id"
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "role": r["role"],
            "content": r["content"],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
        }
        for r in rows
    ]


def clear_conversation(*, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    conn = _connect(db_path)
    conn.execute("DELETE FROM conversation_history")
    conn.commit()
    conn.close()


# ── User overrides ──────────────────────────────────────────────────

def save_override(
    *,
    message_id: str | None = None,
    machine_id: str | None = None,
    date: str | None = None,
    hour_window: str | None = None,
    product: str,
    original_rec: int,
    adjusted_rec: int,
    reason: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    conn = _connect(db_path)
    cur = conn.execute(
        "INSERT INTO user_overrides "
        "(message_id, machine_id, date, hour_window, product, original_rec, adjusted_rec, reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (message_id, machine_id, date, hour_window, product, original_rec, adjusted_rec, reason),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_overrides_for_message(
    message_id: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return override rows tied to one chat message (chronological order).

    Used to reconstruct the user's confirmed quantities after a page refresh when
    session state is cleared but SQLite still holds per-product adjustments.
    """
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT machine_id, product, original_rec, adjusted_rec, reason, hour_window, date "
        "FROM user_overrides WHERE message_id = ? ORDER BY id ASC",
        (message_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_overrides_for_context(
    machine_id: str,
    hour_window: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return recent overrides for a machine + time window (last 30 days).

    Used by the forecast tool to show the agent what the user has
    previously adjusted so it can incorporate those preferences.
    """
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT product, original_rec, adjusted_rec, reason, date "
        "FROM user_overrides "
        "WHERE machine_id = ? AND hour_window = ? "
        "  AND timestamp >= datetime('now', '-30 days') "
        "ORDER BY timestamp DESC",
        (machine_id, hour_window),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── User feedback ───────────────────────────────────────────────────

def save_feedback(
    message_id: str,
    rating: int,
    comment: str | None = None,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    conn = _connect(db_path)
    cur = conn.execute(
        "INSERT INTO user_feedback (message_id, rating, comment) VALUES (?, ?, ?)",
        (message_id, rating, comment),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_feedback_comments(
    limit: int = 5,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return recent negative feedback that includes a comment."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT comment, timestamp FROM user_feedback "
        "WHERE rating < 0 AND comment IS NOT NULL AND comment != '' "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_feedback_stats(
    *, db_path: str | Path = DEFAULT_DB_PATH
) -> dict[str, int]:
    """Aggregate thumbs-up / thumbs-down counts for the sidebar metric."""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT "
        "  SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) AS thumbs_up, "
        "  SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) AS thumbs_down "
        "FROM user_feedback"
    ).fetchone()
    conn.close()
    return {
        "thumbs_up": row["thumbs_up"] or 0,
        "thumbs_down": row["thumbs_down"] or 0,
    }
