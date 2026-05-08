from __future__ import annotations

import json

from src.kb.query import (
    find_mentions_by_type_and_keyword,
    get_mentions_by_pmid,
    get_pmids_by_normalized_id,
)
from src.kb.schema import DEFAULT_DB_PATH, init_sqlite_schema
from src.kb.writer import write_pipeline_outputs_to_sqlite
from src.extraction.bc5cdr_pipeline import run_bc5cdr_pipeline


def main():
    db_path = DEFAULT_DB_PATH
    init_sqlite_schema(db_path)

    papers_df, entities_df = run_bc5cdr_pipeline(
        query="BRCA1 breast cancer",
        smoke=True,
    )
    added = write_pipeline_outputs_to_sqlite(papers_df, entities_df, db_path=db_path)
    print(f"OK: ingest smoke added={added}")

    by_pmid = get_mentions_by_pmid("SMOKE001", db_path=db_path)
    by_norm = get_pmids_by_normalized_id("HGNC:1100", db_path=db_path)
    by_type_kw = find_mentions_by_type_and_keyword("Gene", "brca", db_path=db_path)

    print("OK: query by pmid")
    print(json.dumps(by_pmid, ensure_ascii=False, indent=2))
    print("OK: query by normalized_id")
    print(json.dumps(by_norm, ensure_ascii=False, indent=2))
    print("OK: query by type+keyword")
    print(json.dumps(by_type_kw, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

