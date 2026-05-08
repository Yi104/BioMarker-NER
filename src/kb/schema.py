from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "data/processed/kb/biomed_kb.db"


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from: {start}")


def init_sqlite_schema(db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Create SQLite database file and minimal v1 tables if they do not exist.

    Returns the resolved database path string for logging/CLI output.
    """
    path = Path(db_path)
    if not path.is_absolute():
        repo_root = _find_repo_root(Path(__file__).parent)
        path = repo_root / path
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(resolved))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                pmid TEXT PRIMARY KEY,
                title TEXT,
                year TEXT,
                journal TEXT,
                abstract TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_mentions (
                mention_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pmid TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_text TEXT NOT NULL,
                token_start INTEGER,
                token_end INTEGER,
                normalized_id TEXT,
                normalized_text TEXT,
                normalized_source TEXT,
                normalized_score REAL,
                UNIQUE(pmid, entity_type, entity_text, token_start, token_end)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS normalized_entities (
                normalized_id TEXT PRIMARY KEY,
                preferred_label TEXT,
                entity_type TEXT,
                source_vocab TEXT
            )
            """
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_mentions_pmid ON entity_mentions(pmid)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_mentions_norm_id ON entity_mentions(normalized_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_mentions_type ON entity_mentions(entity_type)"
        )
        conn.commit()
    finally:
        conn.close()

    return str(resolved)
