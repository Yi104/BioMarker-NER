"""
Fine-tune BioBERT for Named Entity Recognition (NER).
"""

import json
import os
import numpy as np
import argparse
import csv
from datetime import datetime
from dataclasses import dataclass, fields
from typing import Optional

# Huggingface
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

# seqeval https://github.com/chakki-works/seqeval，
# sequence labeling evaluation.
# entity-level. for BIO/BILOU (exact span match and entity type match)
from seqeval.metrics import precision_score, recall_score, f1_score

# helpers
from src.extraction.data import load_ner_dataset, tokenize_and_align_labels
from src.extraction.train_utils import set_seed

# Layer: extraction
# Role: train and evaluate the NER model, then export artifacts for inference.

@dataclass
class Config:
    model_name : str
    dataset: str
    max_length: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    warmup_ratio: float
    logging_steps: int
    eval_strategy: str  # epochs
    eval_steps: int
    save_strategy: str # epochs
    save_steps: int
    seed: int
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    checkpoints_dir: str = "outputs/checkpoints/default"
    best_model_dir: str = "outputs/best_model"
    report_path: Optional[str] = None


def _load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        raw_cfg = json.load(f)
    allowed = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in raw_cfg.items() if k in allowed}
    cfg = Config(**filtered)
    # If report_path is omitted, route reports by dataset to avoid ambiguous "default" folders.
    if not cfg.report_path:
        cfg.report_path = f"outputs/reports/{cfg.dataset}/test_metrics.json"
    return cfg


def compute_metrics(eval_pred, label_list):
    """
    Compute evaluation metrics for NER (entity-level).
    convert predictions & labels back to strings, then compute seqeval metrics (entity-level).
    :param eval_pred: Tuple[np.ndarray, np.ndarray]
                     - predictions: model outputs (logits), shape (batch_size, seq_len, num_labels)
                        - labels: true labels, shape (batch_size, seq_len)
    :param label_list: List[str], list of label strings, e.g ["O", "B-Chemical", "I-Chemical", "B-Disease" ...]
    :return: metrics  dict {precision, recall, f1_score}
    """
    predictions, labels = eval_pred

    # Greedy token prediction from logits.
    preds = np.argmax(predictions, axis=2)

    # Convert IDs back to label strings, ignore special tokens (-100)
    true_preds = [
        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(preds, labels)
    ]

    # Compute entity-level metrics
    p = precision_score(true_labels, true_preds)
    r = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)

    return {"precision": p, "recall": r, "f1": f1}


def _save_log_history(log_history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "train_log_history.json")
    with open(json_path, "w") as f:
        json.dump(log_history, f, indent=2)

    # Flatten to CSV-friendly rows
    keys = set()
    for row in log_history:
        keys.update(row.keys())
    ordered = ["step", "epoch", "loss", "eval_loss", "eval_precision", "eval_recall", "eval_f1"]
    remaining = sorted(k for k in keys if k not in ordered)
    columns = ordered + remaining

    csv_path = os.path.join(out_dir, "train_log_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in log_history:
            writer.writerow({k: row.get(k) for k in columns})


def _append_experiment_row(cfg: Config, test_metrics: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "experiment_metrics.csv")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": cfg.model_name,
        "dataset": cfg.dataset,
        "max_length": cfg.max_length,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "num_train_epochs": cfg.num_train_epochs,
        "train_batch_size": cfg.per_device_train_batch_size,
        "eval_batch_size": cfg.per_device_eval_batch_size,
        "seed": cfg.seed,
        "test_loss": test_metrics.get("eval_loss"),
        "test_precision": test_metrics.get("eval_precision"),
        "test_recall": test_metrics.get("eval_recall"),
        "test_f1": test_metrics.get("eval_f1"),
        "test_runtime": test_metrics.get("eval_runtime"),
    }
    columns = list(row.keys())
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _plot_training_curves(log_history, out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skip plotting curves.")
        return

    train_points = [x for x in log_history if "loss" in x and "eval_loss" not in x and "step" in x]
    eval_points = [x for x in log_history if "eval_loss" in x and "step" in x]
    if not train_points and not eval_points:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.ravel()

    if train_points:
        ax[0].plot([x["step"] for x in train_points], [x["loss"] for x in train_points], marker="o", linewidth=1)
    ax[0].set_title("Train Loss")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Loss")

    if eval_points:
        ax[1].plot([x["step"] for x in eval_points], [x["eval_loss"] for x in eval_points], marker="o", linewidth=1)
    ax[1].set_title("Eval Loss")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Loss")

    if eval_points and any("eval_f1" in x for x in eval_points):
        f1_points = [x for x in eval_points if "eval_f1" in x]
        ax[2].plot([x["step"] for x in f1_points], [x["eval_f1"] for x in f1_points], marker="o", linewidth=1)
    ax[2].set_title("Eval F1")
    ax[2].set_xlabel("Step")
    ax[2].set_ylabel("F1")

    if eval_points and any("eval_precision" in x and "eval_recall" in x for x in eval_points):
        pr_points = [x for x in eval_points if "eval_precision" in x and "eval_recall" in x]
        ax[3].plot([x["step"] for x in pr_points], [x["eval_precision"] for x in pr_points], marker="o", label="Precision", linewidth=1)
        ax[3].plot([x["step"] for x in pr_points], [x["eval_recall"] for x in pr_points], marker="o", label="Recall", linewidth=1)
        ax[3].legend()
    ax[3].set_title("Eval Precision/Recall")
    ax[3].set_xlabel("Step")
    ax[3].set_ylabel("Score")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

# ======================= Main Training Pipeline =====================

def main(config_path: str = "configs/bc5cdr.json", eval_only: bool = False):
    """
    Main training pipeline for BioBERT NER.

    Parameters
    ----------
    config_path : str, optional
        Path to JSON configuration file (default "configs/bc5cdr.json").

    Workflow
    --------
    1. Load config and set random seed.
    2. Load dataset and label space.
    3. Tokenize input text and align labels to subwords.
    4. Initialize BioBERT model for token classification.
    5. Define training arguments and Trainer.
    6. Train and evaluate model.
    7. Save best checkpoint and final test metrics.
    """

    # 1) Config and reproducibility
    cfg = _load_config(config_path)
    set_seed(cfg.seed)

    # 2) Dataset loading
    ds, text_col, label_col, label_list = load_ner_dataset(cfg.dataset)

    # 3) Feature construction
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenized = tokenize_and_align_labels(
        ds, tokenizer, text_col, label_col, cfg.max_length
    )


    # 4) Model init + label mapping persistence
    if eval_only:
        model = AutoModelForTokenClassification.from_pretrained(cfg.best_model_dir)
    else:
        model_cfg = AutoConfig.from_pretrained(cfg.model_name)
        model_cfg.num_labels = len(label_list)
        # Normalize project-level names to transformer config keys.
        model_cfg.hidden_dropout_prob = cfg.hidden_dropout
        model_cfg.attention_probs_dropout_prob = cfg.attention_dropout

        model = AutoModelForTokenClassification.from_pretrained(
            cfg.model_name,
            config=model_cfg,
        )
        model.config.label2id = {label: idx for idx, label in enumerate(label_list)}
        model.config.id2label = {idx: label for idx, label in enumerate(label_list)}


    # 5) Trainer configuration
    args = TrainingArguments(
        output_dir=cfg.checkpoints_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=2,            # keep last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="f1",    # choose F1 for best model
        greater_is_better=True,
    )


    # 6) Trainer assembly
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def _compute(eval_pred):
        return compute_metrics(eval_pred, label_list)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute,
    )

    # 7) Fit
    if not eval_only:
        trainer.train()

        # Persist model/tokenizer for app usage.
        trainer.save_model(cfg.best_model_dir)
        tokenizer.save_pretrained(cfg.best_model_dir)


    # 8) Final evaluation report
    eval_split = "test" if "test" in tokenized else "validation"
    if eval_split != "test":
        print("WARN: no 'test' split found; using 'validation' for final evaluation.")
    metrics = trainer.evaluate(tokenized[eval_split])
    metrics["final_eval_split"] = eval_split
    report_dir = os.path.dirname(cfg.report_path)
    os.makedirs(report_dir, exist_ok=True)
    with open(cfg.report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    _save_log_history(trainer.state.log_history, report_dir)
    _append_experiment_row(cfg, metrics, report_dir)
    _plot_training_curves(trainer.state.log_history, report_dir)



def dry_run(config_path: str = "configs/bc5cdr.json"):
    # Fast check for config wiring without launching training.
    cfg = _load_config(config_path)
    print(f"OK: train_ner dry_run config={config_path}")
    print(f"model_name={cfg.model_name} dataset={cfg.dataset} max_length={cfg.max_length}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BioBERT NER.")
    parser.add_argument("--config_path", type=str, default="configs/bc5cdr.json")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate config wiring, do not train.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and run final evaluation on saved best_model_dir.",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.config_path)
    else:
        if args.eval_only:
            print("Running final evaluation only...")
        else:
            print("Training BioBERT on biomedical NER...")
        main(args.config_path, eval_only=args.eval_only)
