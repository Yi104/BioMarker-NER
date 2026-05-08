import argparse

from src.extraction.jnlpba_pipeline import run_jnlpba_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the JNLPBA workflow.")
    parser.add_argument("--query", type=str, default="IL-2 gene expression")
    parser.add_argument("--model_path", type=str, default="outputs/best_model_jnlpba")
    parser.add_argument("--retmax", type=int, default=20)
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
    print(f"OK: JNLPBA workflow mode={mode} papers={len(papers_df)} entities={len(entities_df)}")
