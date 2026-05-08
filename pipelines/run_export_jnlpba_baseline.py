from __future__ import annotations

import argparse
import os

from src.extraction.jnlpba_pipeline import run_jnlpba_pipeline


def main():
    parser = argparse.ArgumentParser(description="Export JNLPBA baseline CSV files.")
    parser.add_argument("--query", type=str, default="IL-2 gene expression")
    parser.add_argument("--retmax", type=int, default=5)
    parser.add_argument("--model_path", type=str, default="outputs/best_model_jnlpba")
    parser.add_argument("--outdir", type=str, default="outputs/reports")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use deterministic smoke mode for local baseline export.",
    )
    args = parser.parse_args()

    papers, entities = run_jnlpba_pipeline(
        query=args.query,
        retmax=args.retmax,
        model_path=args.model_path,
        smoke=args.smoke,
    )

    os.makedirs(args.outdir, exist_ok=True)
    papers_path = os.path.join(args.outdir, "baseline_jnlpba_papers.csv")
    entities_path = os.path.join(args.outdir, "baseline_jnlpba_entities.csv")
    papers.to_csv(papers_path, index=False)
    entities.to_csv(entities_path, index=False)

    mode = "smoke" if args.smoke else "live"
    print(f"OK: saved mode={mode} papers={len(papers)} entities={len(entities)}")
    print(f"papers_csv={papers_path}")
    print(f"entities_csv={entities_path}")
    print(f"papers_cols={list(papers.columns)}")
    print(f"entities_cols={list(entities.columns)}")


if __name__ == "__main__":
    main()

