"""
Dataset loading and token alignment for biomarker NER
This module does:
1. loading biomedical NER datasets (BC5CDR, JNLPBA) from huggingface BigBio
2. Converting their KB schema into flat token-level BIO labels
3. Tokenizing + aligning tokens with labels for modl training.

from huggingface bigbio/bc5cdr:
bc5cdr.py: https://huggingface.co/datasets/bigbio/bc5cdr/blob/main/bc5cdr.py
BUILDER_CONFIGS = [
        BigBioConfig(
            name="bc5cdr_source",
            version=SOURCE_VERSION,
            description="BC5CDR source schema",
            schema="source",
            subset_id="bc5cdr",
        ),
        BigBioConfig(
            name="bc5cdr_bigbio_kb",
            version=BIGBIO_VERSION,
            description="BC5CDR simplified BigBio schema",
            schema="bigbio_kb",
            subset_id="bc5cdr",
        ),


kb_schema: see bigbiohub.py kb_features  https://huggingface.co/datasets/bigbio/bc5cdr/blob/main/bigbiohub.py
each has:
{
passages: the text of abstract or document
entitles: id,text, type, offsets
events:
coreferences
relations
}
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import argparse

# Layer: extraction
# Role: convert raw BigBio examples into token-level training features for NER.

LABEL_ALL_TOKENS = True

# Available datasets
NER_DATASETS = {
    "jnlpba": {
        "path": "bigbio/jnlpba",
        "name": "jnlpba_bigbio_kb",
        "schema": "kb_jnlpba",
        "text_column": "tokens",
        "label_column": "ner_tags",
    },
    "bc5cdr": {
        "path": "bigbio/bc5cdr",
        "name": "bc5cdr_bigbio_kb",   # BigBio schema
        "schema": "kb",
        "text_column": "tokens",      #
        "label_column": "ner_tags",   #
    },
}


def load_ner_dataset(name: str, tokenizer_name: str = None, cache_dir: str = None):
    """
    Load a biomedical dataset in BigBio KB schema and convert entities into BIO labels.

    Args:
        name (str): dataset key ("jnlpba" or "bc5cdr")
        tokenizer_name (str): HuggingFace tokenizer (needed to split text into tokens)
        cache_dir (str, optional): local cache directory

    Returns:
        ds (DatasetDict): HuggingFace DatasetDict with train/valid/test splits.
            Each example has:
                - "tokens": list[str], tokenized words
                - "ner_tags": list[int], label IDs for each token (BIO format)
        text_column (str): name of the text column ("tokens")
        label_column (str): name of the label column ("ner_tags")
        labels (list[str]): list of label names (e.g., ["O", "B-Chemical", "I-Chemical", "B-Disease", ...])
    """
    spec = NER_DATASETS[name]

    # Load raw BigBio KB schema dataset
    ds = load_dataset(
        spec["path"],
        name=spec["name"],
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    if spec["schema"] == "source":
        # jnlpba_source already has token-level ner_tags.
        feature = ds["train"].features[spec["label_column"]]
        labels = None
        if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
            labels = list(feature.feature.names)
        if not labels:
            seen = set()
            for split in ds.keys():
                for seq in ds[split][spec["label_column"]]:
                    seen.update(seq)
            labels = [str(i) for i in sorted(seen)]
    else:
        # Step 1: Convert KB-style entity annotations into BIO tags over whitespace tokens.
        def kb_to_bio(example):
            entities = example["entities"]

            if spec["schema"] == "kb_jnlpba":
                # JNLPBA BigBio KB often has empty passages; entities carry token stream.
                tokens = [ent["text"][0] for ent in entities if ent.get("text")]
                tags = []
                for ent in entities:
                    ent_type = str(ent.get("type", "0"))
                    if ent_type == "0":
                        tags.append("O")
                    else:
                        tags.append(f"B-{ent_type}")
                return {"tokens": tokens, "ner_tags_str": tags}

            text = " ".join(p for passage in example["passages"] for p in passage["text"])
            tokens = text.split()
            tags = ["O"] * len(tokens)
            for ent in entities:
                ent_type = ent["type"]
                for i, tok in enumerate(tokens):
                    if tok == ent["text"][0]:
                        tags[i] = f"B-{ent_type}"
                        for j in range(1, len(ent["text"])):
                            if i + j < len(tokens):
                                tags[i + j] = f"I-{ent_type}"

            return {"tokens": tokens, "ner_tags_str": tags}

        ds = ds.map(kb_to_bio)

        # Step 2: Build deterministic label vocabulary for model config.
        unique_tags = {"O"}
        for split in ds.keys():
            for ex in ds[split]["ner_tags_str"]:
                unique_tags.update(ex)
        labels = sorted(unique_tags)
        label2id = {l: i for i, l in enumerate(labels)}

        # Step 3: Convert string tags into integer IDs expected by Trainer.
        def encode_tags(example):
            return {"ner_tags": [label2id[tag] for tag in example["ner_tags_str"]]}

        ds = ds.map(encode_tags)

    # Keep training code stable: always provide a validation split.
    if "validation" not in ds and "train" in ds:
        split = ds["train"].train_test_split(test_size=0.1, seed=42)
        ds = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
                **({"test": ds["test"]} if "test" in ds else {}),
            }
        )

    return ds, spec["text_column"], spec["label_column"], labels


def tokenize_and_align_labels(ds, tokenizer: AutoTokenizer, text_col: str, label_col: str,
                              max_length: int, label_all_tokens: bool = LABEL_ALL_TOKENS):
    """
    Tokenize dataset and align BIO labels with subword tokens.

    Args:
        ds (DatasetDict): dataset with "tokens" and "ner_tags"
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        text_col (str): text column ("tokens")
        label_col (str): label column ("ner_tags")
        max_length (int): max sequence length
        label_all_tokens (bool): whether to copy label to all subword pieces or not

    Returns:
        tokenized_ds (DatasetDict): dataset with:
            - input_ids
            - attention_mask
            - labels
    """
    def _align(batch):
        # Align word-level labels to subword tokens and mask specials with -100.
        tokenized = tokenizer(batch[text_col], is_split_into_words=True,
                              truncation=True, padding=False, max_length=max_length)
        new_labels = []
        for i, labels in enumerate(batch[label_col]):
            word_ids = tokenized.word_ids(i)
            prev_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(labels[word_id] if label_all_tokens else -100)
                prev_word_id = word_id
            new_labels.append(label_ids)
        tokenized["labels"] = new_labels
        return tokenized

    return ds.map(_align, batched=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset/label alignment smoke check.")
    parser.add_argument("--dataset", type=str, default="bc5cdr")
    parser.add_argument("--model_name", type=str, default="dmis-lab/biobert-base-cased-v1.1")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    ds, text_col, label_col, labels = load_ner_dataset(args.dataset, tokenizer_name=args.model_name)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = tokenize_and_align_labels(ds, tok, text_col, label_col, args.max_length)
    print(f"OK: extraction.data labels={len(labels)} train_rows={len(ds['train'])}")
    print(f"tokenized_columns={tokenized['train'].column_names}")
