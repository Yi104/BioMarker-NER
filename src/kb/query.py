from __future__ import annotations

import sqlite3
from typing import List, Dict

from src.kb.schema import DEFAULT_DB_PATH, init_sqlite_schema


def _rows_to_dicts(cursor: sqlite3.Cursor) -> List[Dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_mentions_by_pmid(pmid: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    resolved_db_path = init_sqlite_schema(db_path)
    conn = sqlite3.connect(resolved_db_path)
    try:
        cur = conn.execute(
            """
            SELECT pmid, entity_type, entity_text, token_start, token_end,
                   normalized_id, normalized_text, normalized_source, normalized_score
            FROM entity_mentions
            WHERE pmid = ?
            ORDER BY token_start, token_end
            """,
            (pmid,),
        )
        return _rows_to_dicts(cur)
    finally:
        conn.close()


def get_pmids_by_normalized_id(normalized_id: str, db_path: str = DEFAULT_DB_PATH) -> List[str]:
    resolved_db_path = init_sqlite_schema(db_path)
    conn = sqlite3.connect(resolved_db_path)
    try:
        cur = conn.execute(
            """
            SELECT DISTINCT pmid
            FROM entity_mentions
            WHERE normalized_id = ?
            ORDER BY pmid
            """,
            (normalized_id,),
        )
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def find_mentions_by_type_and_keyword(
    entity_type: str,
    keyword: str,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict]:
    resolved_db_path = init_sqlite_schema(db_path)
    conn = sqlite3.connect(resolved_db_path)
    try:
        cur = conn.execute(
            """
            SELECT pmid, entity_type, entity_text, normalized_id, normalized_text
            FROM entity_mentions
            WHERE lower(entity_type) = lower(?)
              AND (
                    lower(entity_text) LIKE '%' || lower(?) || '%'
                 OR lower(normalized_text) LIKE '%' || lower(?) || '%'
              )
            ORDER BY pmid
            """,
            (entity_type, keyword, keyword),
        )
        return _rows_to_dicts(cur)
    finally:
        conn.close()

