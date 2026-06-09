import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
) 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from logging_utils import archive_existing_output_txts, make_log_path
from lexiconfeatures import LexiconFeatures
from preprocess import clean_text
from sklearn.preprocessing import FunctionTransformer

LABEL_MAPS = {
    "davidson": {0: "hate_speech", 1: "offensive_language", 2: "neither"},
    "hatexplain": {
        "hatespeech": "hate_speech",
        "hate":       "hate_speech",
        "offensive":  "offensive_language",
        "normal":     "neither",
        "neither":    "neither",
    },
}

LABEL_ORDER = ["hate_speech", "offensive_language", "neither"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True)
    parser.add_argument("--text-col",    default="tweet")
    parser.add_argument("--label-col",   default="class")
    parser.add_argument("--label-map",   choices=["davidson","hatexplain","none"],
                        default="davidson")
    parser.add_argument("--test-size",   type=float, default=0.15)
    parser.add_argument("--dev-size",    type=float, default=0.15)
    parser.add_argument("--random-state",type=int,   default=42)
    parser.add_argument("--models-dir",  default="models")
    parser.add_argument("--figures-dir", default="reports/figures")
    return parser.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_dataset(path, text_col, label_col, label_map_name):
    df = pd.read_csv(path)
    missing = [c for c in [text_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[[text_col, label_col]].dropna().copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    if label_map_name != "none":
        df["label"] = df["label"].map(LABEL_MAPS[label_map_name])

    df = df[df["label"].isin(LABEL_ORDER)].copy()
    if df.empty:
        raise ValueError("No usable rows after label mapping.")

    df["clean_text"] = df["text"].apply(clean_text)
    return df


def split_dataset(df, test_size, dev_size, random_state):
    train_df, temp_df = train_test_split(
        df, test_size=test_size + dev_size,
        random_state=random_state, stratify=df["label"],
    )
    rel_test = test_size / (test_size + dev_size)
    dev_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        random_state=random_state, stratify=temp_df["label"],
    )
    return train_df, dev_df, test_df


# ── Models ────────────────────────────────────────────────────────────────────

def build_models(random_state):
    # Fix 2: LexiconFeatures added to FeatureUnion
    shared_features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=2, max_df=0.95, sublinear_tf=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, sublinear_tf=True,
        )),
        ("lexicon", LexiconFeatures()),  # ← Fix 2
    ])

    # Naive Bayes — plain sklearn Pipeline (no oversampling; NB is less
    # sensitive to imbalance with class priors)
    nb_model = Pipeline([
        ("vec", CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    # Logistic Regression — Fix 3: imblearn Pipeline with RandomOverSampler
    logreg_model = ImbPipeline([
        ("features",  shared_features),
        ("oversample", RandomOverSampler(      # ← Fix 3
            sampling_strategy={"hate_speech": 1500},
            random_state=random_state,
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            # class_weight="balanced",
            random_state=random_state,
        )),
    ])

    return {"naive_bayes_count": nb_model, "logreg_word_char_tfidf": logreg_model}


# ── Threshold tuning (Fix 1) ──────────────────────────────────────────────────

def find_best_hate_threshold(model, texts, labels):
    """
    Sweep probability thresholds on the dev set and pick the one that
    maximises hate_speech F1 without collapsing overall macro F1.
    """
    if not hasattr(model, "predict_proba"):
        print("Model has no predict_proba — skipping threshold search.")
        return 0.5

    probs   = model.predict_proba(texts)
    classes = list(model.classes_)
    hate_idx = classes.index("hate_speech")

    best_t, best_f1 = 0.5, 0.0
    print("\nThreshold sweep (hate_speech F1):")
    for t in np.arange(0.10, 0.55, 0.02):
        preds = []
        for row in probs:
            if row[hate_idx] >= t:
                preds.append("hate_speech")
            else:
                preds.append(classes[int(row.argmax())])
        hs_f1 = f1_score(labels, preds, labels=["hate_speech"], average="macro",
                         zero_division=0)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        marker = " ←" if hs_f1 > best_f1 else ""
        print(f"  t={t:.2f}  hate_F1={hs_f1:.4f}  macro_F1={macro_f1:.4f}{marker}")
        if hs_f1 > best_f1:
            best_f1 = hs_f1
            best_t  = t

    print(f"\nSelected threshold: {best_t:.2f}  (hate_speech F1={best_f1:.4f})")
    return best_t


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(name, model, split_name, texts, labels, threshold=0.5):
    if hasattr(model, "predict_proba") and threshold != 0.5:
        probs    = model.predict_proba(texts)
        classes  = list(model.classes_)
        hate_idx = classes.index("hate_speech")
        preds = [
            "hate_speech" if row[hate_idx] >= threshold
            else classes[int(row.argmax())]
            for row in probs
        ]
    else:
        preds = model.predict(texts)

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"\n=== {name} on {split_name} ===")
    print(f"Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")
    print(classification_report(labels, preds, labels=LABEL_ORDER, digits=4,
                                zero_division=0))
    return preds, {"accuracy": acc, "macro_f1": macro_f1}


def save_confusion_matrix(labels, preds, path):
    cm = confusion_matrix(labels, preds, labels=LABEL_ORDER)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()




def scale_lexicon(X):
    return X * 13.0  # multiply lexicon features to increase influence

    shared_features = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(...)),
    ("char_tfidf", TfidfVectorizer(...)),
    ("lexicon", Pipeline([              # ← wrap in a mini pipeline to scale
        ("feats",  LexiconFeatures()),
        ("scale",  FunctionTransformer(scale_lexicon)),
    ])),
])

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    archived_logs = archive_existing_output_txts(project_root)
    models_dir  = Path(args.models_dir)
    figures_dir = Path(args.figures_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data, args.text_col, args.label_col, args.label_map)
    print("\nClass distribution:\n", df["label"].value_counts())

    train_df, dev_df, test_df = split_dataset(
        df, args.test_size, args.dev_size, args.random_state
    )

    models     = build_models(args.random_state)
    dev_scores = {}

    for name, model in models.items():
        model.fit(train_df["clean_text"], train_df["label"])
        _, scores = evaluate_model(name, model, "dev",
                                   dev_df["clean_text"], dev_df["label"])
        dev_scores[name] = scores["macro_f1"]

    best_name  = max(dev_scores, key=dev_scores.get)
    best_model = models[best_name]
    print(f"\nBest model by dev macro F1: {best_name}")

    # Fix 1: find optimal threshold on dev set
    best_threshold = find_best_hate_threshold(
        best_model, dev_df["clean_text"], dev_df["label"]
    )

    # Final evaluation on test set using the tuned threshold
    test_preds, test_scores = evaluate_model(
        best_name, best_model, "test",
        test_df["clean_text"], test_df["label"],
        threshold=best_threshold,
    )

    # Fix 4: save model + threshold together as a bundle
    model_path = models_dir / "best_model.joblib"
    joblib.dump({"model": best_model, "hate_threshold": best_threshold}, model_path)
    print(f"\nSaved bundle: {model_path}  (threshold={best_threshold:.2f})")

    cm_path = figures_dir / "confusion_matrix.png"
    save_confusion_matrix(test_df["label"], test_preds, cm_path)
    print(f"Saved confusion matrix: {cm_path}")

    results_path = make_log_path(project_root, "results")
    with results_path.open("w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Hate speech threshold: {best_threshold:.2f}\n")
        f.write(f"Test accuracy: {test_scores['accuracy']:.4f}\n")
        f.write(f"Test macro F1: {test_scores['macro_f1']:.4f}\n\n")
        f.write(classification_report(test_df["label"], test_preds,
                                      labels=LABEL_ORDER, zero_division=0))
        if archived_logs:
            f.write("\n\nArchived previous output txt files:\n")
            for archived_log in archived_logs:
                f.write(f"{archived_log}\n")
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
