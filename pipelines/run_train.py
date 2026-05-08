import argparse

from src.extraction.train_ner import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER training pipeline.")
    parser.add_argument("--config_path", type=str, default="configs/bc5cdr.json")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and run final evaluation on saved best_model_dir.",
    )
    args = parser.parse_args()
    main(args.config_path, eval_only=args.eval_only)
