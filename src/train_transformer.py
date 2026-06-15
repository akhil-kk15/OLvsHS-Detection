"""Train a transformer classifier for the HSvOL project."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from logging_utils import archive_existing_output_txts, make_log_path
from preprocess import clean_text


# Keep this aligned with src/train.py so the transformer uses the same labels.
LABEL_ORDER = ["hate_speech", "offensive_language", "neither"]
LABEL_MAPS = {
    "davidson": {0: "hate_speech", 1: "offensive_language", 2: "neither"},
    "hatexplain": {
        "hatespeech": "hate_speech",
        "hate": "hate_speech",
        "offensive": "offensive_language",
        "normal": "neither",
        "neither": "neither",
    },
}

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


class EncodedTextDataset(Dataset):
    """Store pre-tokenized inputs and integer labels for Hugging Face Trainer."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def parse_args():
    """Parse CLI arguments for transformer training."""

    parser = argparse.ArgumentParser(description="Train a transformer classifier.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--text-col", default="tweet")
    parser.add_argument("--label-col", default="class")
    parser.add_argument(
        "--label-map",
        choices=["davidson", "hatexplain", "none"],
        default="davidson",
    )
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="models/distilbert_hsvol")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--dev-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--external-gate-data",
        default="data/external_gate_project_mapped.csv",
        help="Optional extra three-class rows for training.",
    )
    parser.add_argument(
        "--external-severity-data",
        default="data/external_severity_project_mapped.csv",
        help="Optional extra hate/offensive rows for training.",
    )
    parser.add_argument(
        "--calibration-data",
        default="data/manual_calibration_project_mapped.csv",
        help="Optional manually calibrated rows for training only.",
    )
    parser.add_argument(
        "--calibration-weight",
        type=int,
        default=10,
        help="Repeat calibration rows this many times when enabled.",
    )
    parser.add_argument(
        "--no-external-data",
        action="store_true",
        help="Skip the external gate/severity training files.",
    )
    parser.add_argument(
        "--no-calibration-data",
        action="store_true",
        help="Skip the manually calibrated training rows.",
    )
    return parser.parse_args()


def resolve_path(project_root, raw_path):
    """Resolve a CLI path relative to the project root when needed."""

    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def read_table(path):
    """Read CSV/TSV files while handling the BOM used by some exported CSVs."""

    path = Path(path)
    kwargs = {"encoding": "utf-8-sig"}
    if path.suffix.lower() in {".tsv", ".txt"}:
        kwargs["sep"] = "\t"
    return pd.read_csv(path, **kwargs)


def load_dataset(path, text_col, label_col, label_map_name):
    """
    Load a labeled dataset and map it to the project label order.

    The main Davidson-style dataset is mapped from numeric labels.
    External/calibration datasets already use final label names.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = read_table(path)
    missing = [c for c in [text_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[[text_col, label_col]].dropna().copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    if label_map_name != "none":
        if label_map_name == "davidson":
            label_source = pd.to_numeric(df["label"], errors="coerce")
        else:
            label_source = df["label"].astype(str)
        df["label"] = label_source.map(LABEL_MAPS[label_map_name])

    df = df[df["label"].isin(LABEL_ORDER)].copy()
    if df.empty:
        raise ValueError(f"No usable rows remain after label mapping for {path}.")

    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].apply(clean_text)
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)
    return df[["text", "clean_text", "label", "label_id"]].reset_index(drop=True)


def repeat_frame(df, weight):
    """Repeat rows to up-weight a small curated calibration set."""

    if df.empty or weight <= 1:
        return df.reset_index(drop=True)
    return pd.concat([df] * weight, ignore_index=True)


def split_dataset(df, test_size, dev_size, random_state):
    """Create train/dev/test splits without leaking rows across splits."""

    if test_size <= 0 or dev_size <= 0 or test_size + dev_size >= 1:
        raise ValueError("test-size and dev-size must be positive and sum to < 1.")

    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + dev_size,
        random_state=random_state,
        stratify=stratify,
    )

    rel_test = test_size / (test_size + dev_size)
    temp_stratify = temp_df["label"] if temp_df["label"].value_counts().min() >= 2 else None
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        random_state=random_state,
        stratify=temp_stratify,
    )

    return (
        train_df.reset_index(drop=True),
        dev_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_metrics(y_true, y_pred):
    """Compute accuracy, macro F1, and per-class precision/recall/F1."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for idx, label in enumerate(LABEL_ORDER):
        metrics[f"{label}_precision"] = float(precision[idx])
        metrics[f"{label}_recall"] = float(recall[idx])
        metrics[f"{label}_f1"] = float(f1[idx])
    return metrics


def evaluate_split(trainer, dataset, split_name):
    """Run evaluation and print a readable report for the log files."""

    output = trainer.predict(dataset)
    pred_ids = np.argmax(output.predictions, axis=-1)
    true_ids = output.label_ids
    metrics = build_metrics(true_ids, pred_ids)

    print(f"\n=== Transformer on {split_name} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
    print(
        classification_report(
            true_ids,
            pred_ids,
            labels=list(range(len(LABEL_ORDER))),
            target_names=LABEL_ORDER,
            digits=4,
            zero_division=0,
        )
    )
    return pred_ids, metrics


def load_optional_training_data(path):
    """
    Load optional project CSVs that already use the final text/label schema.

    If the file is missing, return an empty frame so the caller can keep going.
    """

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["text", "clean_text", "label", "label_id"])

    df = read_table(path)
    missing = [c for c in ["text", "label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df = df[df["label"].isin(LABEL_ORDER)].copy()
    if df.empty:
        return pd.DataFrame(columns=["text", "clean_text", "label", "label_id"])

    df["clean_text"] = df["text"].apply(clean_text)
    df["label_id"] = df["label"].map(LABEL2ID).astype(int)
    return df.reset_index(drop=True)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = resolve_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_existing_output_txts(project_root)

    main_df = load_dataset(args.data, args.text_col, args.label_col, args.label_map)
    train_df, dev_df, test_df = split_dataset(
        main_df, args.test_size, args.dev_size, args.random_state
    )

    # Keep the evaluation splits fixed. Only the training rows absorb extras.
    training_parts = [train_df[["text", "clean_text", "label", "label_id"]]]

    if not args.no_external_data:
        external_gate_df = load_optional_training_data(
            resolve_path(project_root, args.external_gate_data)
        )
        external_severity_df = load_optional_training_data(
            resolve_path(project_root, args.external_severity_data)
        )
        if not external_gate_df.empty:
            training_parts.append(external_gate_df)
        if not external_severity_df.empty:
            training_parts.append(external_severity_df)

    if not args.no_calibration_data:
        calibration_df = load_optional_training_data(
            resolve_path(project_root, args.calibration_data)
        )
        if not calibration_df.empty:
            training_parts.append(repeat_frame(calibration_df, args.calibration_weight))

    training_df = pd.concat(training_parts, ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = EncodedTextDataset(
        training_df["clean_text"],
        training_df["label_id"],
        tokenizer,
        args.max_length,
    )
    dev_dataset = EncodedTextDataset(
        dev_df["clean_text"],
        dev_df["label_id"],
        tokenizer,
        args.max_length,
    )
    test_dataset = EncodedTextDataset(
        test_df["clean_text"],
        test_df["label_id"],
        tokenizer,
        args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        # Newer Transformers versions use `eval_strategy` instead of the older
        # `evaluation_strategy` name.
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        seed=args.random_state,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return build_metrics(labels, preds)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    _, dev_metrics = evaluate_split(trainer, dev_dataset, "dev")
    _, test_metrics = evaluate_split(trainer, test_dataset, "test")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "data": str(resolve_path(project_root, args.data)),
        "text_col": args.text_col,
        "label_col": args.label_col,
        "label_map": args.label_map,
        "label_order": LABEL_ORDER,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "test_size": args.test_size,
        "dev_size": args.dev_size,
        "random_state": args.random_state,
        "used_external_data": not args.no_external_data,
        "used_calibration_data": not args.no_calibration_data,
        "external_gate_data": str(resolve_path(project_root, args.external_gate_data)),
        "external_severity_data": str(resolve_path(project_root, args.external_severity_data)),
        "calibration_data": str(resolve_path(project_root, args.calibration_data)),
        "calibration_weight": args.calibration_weight,
        "train_rows": int(len(training_df)),
        "dev_rows": int(len(dev_df)),
        "test_rows": int(len(test_df)),
    }

    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "dev": dev_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    results_path = make_log_path(project_root, "transformer_results")
    with results_path.open("w", encoding="utf-8") as f:
        f.write("Model: transformer\n")
        f.write(f"Base model: {args.model_name}\n")
        f.write(f"Data: {resolve_path(project_root, args.data)}\n")
        f.write(f"Label map: {args.label_map}\n")
        f.write(f"External data: {not args.no_external_data}\n")
        f.write(f"Calibration data: {not args.no_calibration_data}\n")
        f.write(f"Calibration weight: {args.calibration_weight}\n")
        f.write(f"Test accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Test macro F1: {test_metrics['macro_f1']:.4f}\n\n")
        f.write(f"Training rows: {len(training_df)}\n")
        f.write(f"Dev rows: {len(dev_df)}\n")
        f.write(f"Test rows: {len(test_df)}\n\n")
        f.write("Dev metrics:\n")
        f.write(json.dumps(dev_metrics, indent=2))
        f.write("\n\nTest metrics:\n")
        f.write(json.dumps(test_metrics, indent=2))
        f.write("\n")

    print(f"\nSaved transformer model to: {output_dir}")
    print(f"Saved results log to: {results_path}")


if __name__ == "__main__":
    main()
