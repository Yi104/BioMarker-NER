from __future__ import annotations

import argparse

from src.kb.schema import DEFAULT_DB_PATH, init_sqlite_schema


def main():
    parser = argparse.ArgumentParser(description="Initialize SQLite KB schema.")
    parser.add_argument("--db_path", type=str, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    resolved = init_sqlite_schema(args.db_path)
    print(f"OK: initialized sqlite schema at {resolved}")


if __name__ == "__main__":
    main()

