from __future__ import annotations

import pandas as pd

from src.kb.query import (
    find_mentions_by_type_and_keyword,
    get_mentions_by_pmid,
    get_pmids_by_normalized_id,
)
from src.kb.schema import init_sqlite_schema
from src.kb.writer import write_pipeline_outputs_to_sqlite


def test_kb_writer_and_query_roundtrip(tmp_path):
    db_path = str(tmp_path / "biomed_kb_test.db")
    init_sqlite_schema(db_path)

    papers_df = pd.DataFrame(
        [
            {
                "pmid": "P1",
                "title": "Paper 1",
                "year": "2024",
                "journal": "J1",
                "abstract": "BRCA1 and breast cancer",
            }
        ]
    )
    entities_df = pd.DataFrame(
        [
            {
                "pmid": "P1",
                "entity_type": "Gene",
                "entity_text": "BRCA1",
                "token_start": 0,
                "token_end": 0,
                "normalized_id": "HGNC:1100",
                "normalized_text": "BRCA1",
                "normalized_source": "rule_alias_v1",
                "normalized_score": 1.0,
            }
        ]
    )

    added = write_pipeline_outputs_to_sqlite(papers_df, entities_df, db_path=db_path)
    assert added == (1, 1, 1)

    # Idempotency: second write should not duplicate rows.
    added_again = write_pipeline_outputs_to_sqlite(papers_df, entities_df, db_path=db_path)
    assert added_again == (0, 0, 0)

    mentions = get_mentions_by_pmid("P1", db_path=db_path)
    assert len(mentions) == 1
    assert mentions[0]["normalized_id"] == "HGNC:1100"

    pmids = get_pmids_by_normalized_id("HGNC:1100", db_path=db_path)
    assert pmids == ["P1"]

    matches = find_mentions_by_type_and_keyword("Gene", "brca", db_path=db_path)
    assert len(matches) == 1
    assert matches[0]["pmid"] == "P1"

