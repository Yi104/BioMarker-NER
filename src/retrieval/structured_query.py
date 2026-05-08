from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple
import argparse

import pandas as pd

from src.extraction.ner_infer import ner
from src.ingestion.pubmed_client import fetch_pubmed_details, search_pubmed
from src.normalization.rule_based import normalize_entities_df

# Layer: retrieval
# Role: end-to-end query-time path (search -> extract -> tabular outputs).


def run_search_ner_pipeline(
    query: str,
    model_path: str = "outputs/best_model",
    retmax: int = 20,
    max_length: int = 256,
    year_from: int | None = None,
    year_to: int | None = None,
    journal: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the current lightweight retrieval pipeline and return two tables:
    papers summary and extracted entity rows.
    """
    # Step 1: fetch candidate papers from PubMed.
    pmids = search_pubmed(
        query=query,
        retmax=retmax,
        year_from=year_from,
        year_to=year_to,
        journal=journal,
    )
    records = fetch_pubmed_details(pmids)

    # Step 2: run NER per abstract and aggregate outputs.
    rows: List[Dict] = []
    entity_rows: List[Dict] = []
    for rec in records:
        if not rec.abstract:
            continue

        # Current tokenization is whitespace-based for fast iteration.
        tokens = rec.abstract.split()
        pred = ner(model_path=model_path, text_tokens=tokens, max_length=max_length)
        entities = pred["entities"]

        type_counts = Counter(ent["type"] for ent in entities)
        rows.append(
            {
                "pmid": rec.pmid,
                "title": rec.title,
                "year": rec.year,
                "journal": rec.journal,
                "abstract": rec.abstract,
                "entity_count": len(entities),
                "entity_types": ", ".join(
                    f"{k}:{v}" for k, v in sorted(type_counts.items(), key=lambda x: x[0])
                ),
            }
        )

        for ent in entities:
            entity_rows.append(
                {
                    "pmid": rec.pmid,
                    "entity_type": ent["type"],
                    "entity_text": ent["text"],
                    "token_start": ent["start"],
                    "token_end": ent["end"],
                }
            )

    papers_df = pd.DataFrame(rows)
    entities_df = pd.DataFrame(entity_rows)
    # Step 3: apply deterministic normalization layer (v1 rules).
    entities_df = normalize_entities_df(entities_df)
    return papers_df, entities_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured retrieval smoke check.")
    parser.add_argument("--query", type=str, default="BRCA1 breast cancer")
    parser.add_argument("--model_path", type=str, default="outputs/best_model")
    parser.add_argument("--retmax", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    papers_df, entities_df = run_search_ner_pipeline(
        query=args.query,
        model_path=args.model_path,
        retmax=args.retmax,
        max_length=args.max_length,
    )
    print(f"OK: retrieval papers={len(papers_df)} entities={len(entities_df)}")
