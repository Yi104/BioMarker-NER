from __future__ import annotations

import argparse
import json

from src.kb.query import (
    find_mentions_by_type_and_keyword,
    get_mentions_by_pmid,
    get_pmids_by_normalized_id,
)
from src.kb.schema import DEFAULT_DB_PATH, init_sqlite_schema


def main():
    parser = argparse.ArgumentParser(description="Query SQLite KB.")
    parser.add_argument("--db_path", type=str, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--mode",
        choices=["pmid", "normalized_id", "type_keyword"],
        required=True,
    )
    parser.add_argument("--pmid", type=str, default=None)
    parser.add_argument("--normalized_id", type=str, default=None)
    parser.add_argument("--entity_type", type=str, default=None)
    parser.add_argument("--keyword", type=str, default=None)
    args = parser.parse_args()

    init_sqlite_schema(args.db_path)

    if args.mode == "pmid":
        if not args.pmid:
            raise ValueError("--pmid is required when --mode pmid")
        result = get_mentions_by_pmid(args.pmid, db_path=args.db_path)
    elif args.mode == "normalized_id":
        if not args.normalized_id:
            raise ValueError("--normalized_id is required when --mode normalized_id")
        result = get_pmids_by_normalized_id(args.normalized_id, db_path=args.db_path)
    else:
        if not args.entity_type or not args.keyword:
            raise ValueError("--entity_type and --keyword are required when --mode type_keyword")
        result = find_mentions_by_type_and_keyword(
            args.entity_type,
            args.keyword,
            db_path=args.db_path,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

