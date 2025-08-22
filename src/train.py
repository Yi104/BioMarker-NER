"""
Fine-tune BioBERT for Named Entity Recognition (NER)

This script include:
- training pipeline using Huggingface 'Trainer'
- hypermarameter configs via JSON
- Seqeval metrics (f1, precision, recall) at entity level
"""

import json, os
import numpy as np
from dataclasses import dataclass
import pandas as pd
import argparse

# Huggingface
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification,AutoConfig)
# Weight & Bias
import wandb

# seqeval https://github.com/chakki-works/seqeval
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# helpers
from src.data import load_ner_dataset, tokenize_and_align_labels
from src.utils import set_seed


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
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    seed: int
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    run_name: str = "default"


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

    # shape: (batch_size, seq_len, num_labels)
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", type=str, default=None)
    p.add_argument("--model_name", type=str, default="dmis-lab/biobert-base-cased-v1.1")
    p.add_argument("--dataset", type=str, default="bc5cdr")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_strategy", type=str, default="epoch")  # "no"|"steps"|"epoch"
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_strategy", type=str, default="epoch")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dropout", type=float, default=0.1)
    p.add_argument("--attention_dropout", type=float, default=0.1)
    p.add_argument("--run_name", type=str, default="baseline")
    return p.parse_args()
# ======================= Main Training Pipeline =====================

def main(config_path: str = "configs/base.json"):
    """
    Main training pipeline for BioBERT NER.

    Parameters
    ----------
    config_path : str, optional
        Path to JSON configuration file (default "configs/base.json").

    Workflow
    --------
    1. Load config and set random seed.
    2. Load dataset and label space.
    3. Tokenize input text and align labels to subwords.
    4. Initialize BioBERT model for token classification.
    5. Define training arguments
    6. Define Trainer.
    7. Train model.
    8. Evaluate model and Save best checkpoint and final test metrics.
    """

    # ---1. Load configuration---
    # Parse CLI + JSON defaults
    args = parse_args()
    cfg_defaults = {}
    if args.config_path:
        with open(args.config_path) as f:
            cfg_defaults = json.load(f)
    # Merge JSON defaults with CLI
    merged = {**cfg_defaults,
              **{k:v for k, v in vars(args).items() if k != "config_path"}}

    cfg = Config(**merged)
    run = wandb.init( project="biomarker-ner",
                name=cfg.run_name,
                config=cfg.__dict__
         )
    set_seed(cfg.seed)


    # 2. Load dataset
    ds, text_col, label_col, label_list = load_ner_dataset(cfg.dataset)

    # 3. Tokenizer + label alignment
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenized = tokenize_and_align_labels(
        ds, tokenizer, text_col, label_col, cfg.max_length
    )


    # 4. Model (BioBERT for NER)
    model_config = AutoConfig.from_pretrained(
        cfg.model_name,
        num_labels=len(label_list),
        hidden_dropout_prob=cfg.hidden_dropout,
        attention_probs_dropout_prob=cfg.attention_dropout
    )


    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        config = model_config
    )


    # 5. Training arguments
    args = TrainingArguments(
        output_dir="outputs/checkpoints",
        run_name=cfg.run_name,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        warmup_ratio=cfg.warmup_ratio,
        evaluation_strategy=cfg.eval_strategy, # ? evaluation strategy?
        save_strategy=cfg.save_strategy,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=2,            # keep last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="f1",    # choose F1 for best model
        greater_is_better=True,
        report_to = "wandb"
    )


    # 6. Trainer
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

    # 7. Train model
    trainer.train()

    # Save best model checkpoint
    trainer.save_model("outputs/best_model")
    tokenizer.save_pretrained("outputs/best_model")


    # 8. Final test evaluation and save results
    try:
        metrics = trainer.evaluate(tokenized["test"])
    except Exception as e:
        print(f"[WARN] Test evaluation failed: {e}")
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # --- save JSON ---
    os.makedirs("outputs/reports", exist_ok=True)
    run_csv = f'outputs/reports/runs/run_{run.id}.csv'
    pd.DataFrame([{**cfg.__dict__, **metrics}]).to_csv(run_csv, index=False)

    # Append result to master csv
    master_path = "outputs/reports/sweep_all_results.csv"
    if os.path.exists(master_path):
        df = pd.read_csv(master_path)
        df = pd.concat([df, pd.DataFrame([{**cfg.__dict__, **metrics}])], ignore_index=True)
    else:
        df = pd.DataFrame([{**cfg.__dict__, **metrics}])
    df.to_csv(master_path, index=False)
    print(f"[INFO] Results appended to {master_path}")

    # --- log to W&B ---
    wandb.log({"test_metrics": metrics})  # push result to W&B dashboard
    # close w&b run
    wandb.finish()



if __name__ == "__main__":
    print("Training BioBERT on biomedical NER...")
    main()


