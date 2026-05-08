from __future__ import annotations

from src.contracts.task_output_schemas import (
    COMMON_ENTITIES_COLUMNS_V1,
    COMMON_ENTITIES_COLUMNS_V2,
    COMMON_PAPERS_COLUMNS_V1,
)


SCHEMA_REGISTRY: dict[str, dict[str, list[str]]] = {
    "bc5cdr:v1": {
        "papers_columns": COMMON_PAPERS_COLUMNS_V1,
        "entities_columns": COMMON_ENTITIES_COLUMNS_V1,
    },
    "jnlpba:v1": {
        "papers_columns": COMMON_PAPERS_COLUMNS_V1,
        "entities_columns": COMMON_ENTITIES_COLUMNS_V1,
    },
    "bc5cdr:v2": {
        "papers_columns": COMMON_PAPERS_COLUMNS_V1,
        "entities_columns": COMMON_ENTITIES_COLUMNS_V2,
    },
    "jnlpba:v2": {
        "papers_columns": COMMON_PAPERS_COLUMNS_V1,
        "entities_columns": COMMON_ENTITIES_COLUMNS_V2,
    },
}


def get_schema(schema_key: str) -> dict[str, list[str]]:
    if schema_key not in SCHEMA_REGISTRY:
        raise KeyError(f"Unknown schema key: {schema_key}")
    return SCHEMA_REGISTRY[schema_key]
