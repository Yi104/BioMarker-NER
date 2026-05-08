from __future__ import annotations

COMMON_PAPERS_COLUMNS_V1: list[str] = [
    "pmid",
    "title",
    "year",
    "journal",
    "abstract",
    "entity_count",
    "entity_types",
]

COMMON_ENTITIES_COLUMNS_V1: list[str] = [
    "pmid",
    "entity_type",
    "entity_text",
    "token_start",
    "token_end",
]

# v2 extends v1 with normalization-layer outputs.
COMMON_ENTITIES_COLUMNS_V2: list[str] = [
    *COMMON_ENTITIES_COLUMNS_V1,
    "normalized_text",
    "normalized_id",
    "normalized_source",
    "normalized_score",
]
