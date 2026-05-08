from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Layer: normalization
# Role: provide a deterministic v1 normalization pass for entity rows.
#
# Why rule-based first:
# - no external service dependency
# - predictable behavior for regression tests
# - easy to inspect and extend with domain dictionaries later


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from: {start}")


_TYPE_TO_FILE: Dict[str, str] = {
    "gene": "gene_aliases.csv",
    "disease": "disease_aliases.csv",
    "chemical": "chemical_aliases.csv",
}

# Treat protein mentions as gene-family normalization in v1.
_TYPE_FALLBACK: Dict[str, str] = {"protein": "gene"}


def _canonicalize_text(text: str) -> str:
    """
    Normalize surface text into a stable matching key.

    Steps are intentionally simple and deterministic:
    1) lowercase
    2) trim repeated whitespace
    3) remove leading/trailing punctuation noise
    """
    norm = (text or "").strip().lower()
    norm = re.sub(r"\s+", " ", norm)
    norm = norm.strip(".,;:()[]{}\"'")
    return norm


@lru_cache(maxsize=None)
def _load_alias_map_for_type(entity_type: str) -> Dict[str, Tuple[str, str]]:
    """
    Load one entity-type mapping file from data/processed/normalization.
    Cached per entity type to avoid repeated disk reads.
    """
    base_type = _TYPE_FALLBACK.get(entity_type, entity_type)
    filename = _TYPE_TO_FILE.get(base_type)
    if not filename:
        return {}

    repo_root = _find_repo_root(Path(__file__).parent)
    csv_path = repo_root / "data" / "processed" / "normalization" / filename
    if not csv_path.exists():
        return {}

    alias_map: Dict[str, Tuple[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias = _canonicalize_text(row.get("alias", ""))
            norm_id = (row.get("normalized_id") or "").strip()
            preferred = (row.get("preferred_label") or "").strip()
            if alias and norm_id and preferred and alias not in alias_map:
                alias_map[alias] = (norm_id, preferred)
    return alias_map


def _normalize_one(entity_type: str, entity_text: str) -> Tuple[str, str, str, float]:
    """
    Return a normalized tuple:
    (normalized_text, normalized_id, normalized_source, normalized_score)
    """
    normalized_text = _canonicalize_text(entity_text)
    type_key = (entity_type or "").strip().lower()
    alias_map = _load_alias_map_for_type(type_key)

    # Exact alias hit => high confidence deterministic mapping.
    if normalized_text in alias_map:
        norm_id, preferred = alias_map[normalized_text]
        return preferred, norm_id, "rule_alias_v1", 1.0

    # Fallback: keep cleaned surface text, unresolved ID.
    return normalized_text, "UNRESOLVED", "rule_fallback_v1", 0.5


def normalize_entities_df(entities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalization columns to entities table without removing existing fields.

    Output columns added:
    - normalized_text
    - normalized_id
    - normalized_source
    - normalized_score
    """
    out = entities_df.copy()
    if out.empty:
        # Preserve contract even when there are zero entities.
        out["normalized_text"] = pd.Series(dtype="object")
        out["normalized_id"] = pd.Series(dtype="object")
        out["normalized_source"] = pd.Series(dtype="object")
        out["normalized_score"] = pd.Series(dtype="float64")
        return out

    normalized = [
        _normalize_one(str(row.get("entity_type", "")), str(row.get("entity_text", "")))
        for _, row in out.iterrows()
    ]
    out["normalized_text"] = [x[0] for x in normalized]
    out["normalized_id"] = [x[1] for x in normalized]
    out["normalized_source"] = [x[2] for x in normalized]
    out["normalized_score"] = [x[3] for x in normalized]
    return out
