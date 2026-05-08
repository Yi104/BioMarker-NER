from __future__ import annotations

import argparse
from typing import Tuple

import pandas as pd

from src.normalization.rule_based import normalize_entities_df
from src.retrieval.structured_query import run_search_ner_pipeline

# Task B: broader biomedical entity discovery.
# This module mirrors Task A flow but targets the JNLPBA model/output line.


def run_jnlpba_pipeline(
    query: str,
    model_path: str = "outputs/best_model_jnlpba",
    retmax: int = 20,
    max_length: int = 256,
    year_from: int | None = None,
    year_to: int | None = None,
    journal: str | None = None,
    smoke: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    JNLPBA task wrapper for query -> PubMed -> NER -> tables flow.
    """
    if smoke:
        papers_df = pd.DataFrame(
            [
                {
                    "pmid": "SMOKE-JNLPBA-001",
                    "title": "JNLPBA smoke test paper",
                    "year": "2024",
                    "journal": "Smoke Journal",
                    "abstract": "IL-2 regulates T cell activation in human blood.",
                    "entity_count": 3,
                    "entity_types": "Cell_type:1, DNA:1, Protein:1",
                }
            ]
        )
        entities_df = pd.DataFrame(
            [
                {
                    "pmid": "SMOKE-JNLPBA-001",
                    "entity_type": "Protein",
                    "entity_text": "IL-2",
                    "token_start": 0,
                    "token_end": 0,
                },
                {
                    "pmid": "SMOKE-JNLPBA-001",
                    "entity_type": "Cell_type",
                    "entity_text": "T cell",
                    "token_start": 2,
                    "token_end": 3,
                },
                {
                    "pmid": "SMOKE-JNLPBA-001",
                    "entity_type": "DNA",
                    "entity_text": "human blood",
                    "token_start": 6,
                    "token_end": 7,
                },
            ]
        )
        return papers_df, normalize_entities_df(entities_df)

    return run_search_ner_pipeline(
        query=query,
        model_path=model_path,
        retmax=retmax,
        max_length=max_length,
        year_from=year_from,
        year_to=year_to,
        journal=journal,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JNLPBA task workflow.")
    parser.add_argument("--query", type=str, default="IL-2 gene expression")
    parser.add_argument("--model_path", type=str, default="outputs/best_model_jnlpba")
    parser.add_argument("--retmax", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run deterministic local smoke test without network/model dependencies.",
    )
    args = parser.parse_args()

    papers_df, entities_df = run_jnlpba_pipeline(
        query=args.query,
        model_path=args.model_path,
        retmax=args.retmax,
        max_length=args.max_length,
        smoke=args.smoke,
    )
    mode = "smoke" if args.smoke else "live"
    print(f"OK: jnlpba mode={mode} papers={len(papers_df)} entities={len(entities_df)}")
