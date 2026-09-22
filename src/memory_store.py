"""Persistent memory of facts Jarvis knows about the user.

Storage
-------
Facts live in a single-file SQLite database (``data/jarvis_memory.db`` by
default).  Set the ``JARVIS_MEMORY_DB`` environment variable to override the
path -- useful for pointing at a mounted persistent volume when running on
LiveKit Cloud, whose containers are otherwise ephemeral.

The store is safe for concurrent access from multiple threads and from
separate Python processes thanks to SQLite WAL mode.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from livekit.agents import RunContext, function_tool

logger = logging.getLogger("agent.memory")

_DB_NAME = "jarvis_memory.db"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    fact      TEXT    NOT NULL UNIQUE,
    created_at TEXT   NOT NULL
)
"""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _default_db_path() -> Path:
    """Return the default path for the memory database.

    1. ``JARVIS_MEMORY_DB`` env-var (explicit override).
    2. ``data/<repo-root>/jarvis_memory.db`` relative to *this* source file
       so that the path is stable regardless of the working directory.
    """
    override = os.environ.get("JARVIS_MEMORY_DB")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "data" / _DB_NAME


# ---------------------------------------------------------------------------
# Core store
# ---------------------------------------------------------------------------


class MemoryStore:
    """SQLite-backed store of facts Jarvis remembers about the user.

    Parameters
    ----------
    db_path:
        Filesystem path for the SQLite database.  Defaults to the value
        returned by ``_default_db_path()``.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Ensure the schema exists so callers get a consistent store.
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # -- internal helpers ---------------------------------------------------

    def _conn(self) -> closing[sqlite3.Connection]:  # type: ignore[type-arg]
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return closing(conn)

    # -- public API ---------------------------------------------------------

    def remember_fact(self, fact: str) -> bool:
        """Persist *fact* unless an identical one already exists.

        Returns ``True`` if the fact was newly inserted, ``False`` if it
        already existed (deduplicated).  Raises :class:`ValueError` for
        blank strings.
        """
        normalized = _normalize(fact)
        if not normalized:
            raise ValueError("fact must be a non-empty string")

        now = datetime.now(timezone.utc).isoformat()

        with self._lock, self._conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO facts (fact, created_at) VALUES (?, ?)",
                    (normalized, now),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def recall_facts(self) -> list[str]:
        """Return every stored fact, newest first."""
        with self._lock, self._conn() as conn:
            rows = conn.execute("SELECT fact FROM facts ORDER BY id DESC").fetchall()
            return [row["fact"] for row in rows]

    def clear(self) -> None:
        """Delete all stored facts."""
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM facts")
            conn.commit()


def _normalize(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Module-level helpers (used by the agent to seed the initial context)
# ---------------------------------------------------------------------------


def add_fact(fact: str) -> bool:
    """Convenience wrapper around ``MemoryStore().remember_fact``."""
    return MemoryStore().remember_fact(fact)


def list_facts() -> list[str]:
    """Convenience wrapper around ``MemoryStore().recall_facts``."""
    return MemoryStore().recall_facts()


# ---------------------------------------------------------------------------
# LiveKit @function_tool wrappers
# ---------------------------------------------------------------------------


@function_tool
async def remember_fact(context: RunContext, fact: str) -> str:
    """Store a fact about the user so Jarvis can recall it in future conversations.

    Args:
        fact: The fact to remember, e.g. "The user's name is Satvik".
    """
    stored = add_fact(fact)
    return "Got it." if stored else "I already know that."


@function_tool
async def recall_facts(context: RunContext) -> str:
    """Retrieve every fact Jarvis has remembered about the user across sessions."""
    facts = list_facts()
    if not facts:
        return "I have not stored any facts about you yet."
    return "\n".join(f"- {f}" for f in facts)
