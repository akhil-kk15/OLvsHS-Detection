import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

from preprocess import clean_text


LABEL_MAPS = {
    "davidson": {
        0: "hate_speech",
        1: "offensive_language",
        2: "neither",
    },
    "hatexplain": {
        "hatespeech": "hate_speech",
        "hate": "hate_speech",
        "offensive": "offensive_language",
        "normal": "neither",
        "neither": "neither",
    },
}

LABEL_ORDER = ["hate_speech", "offensive_language", "neither"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train hate speech vs offensive language classifiers."
    )
    parser.add_argument("--data", required=True, help="Path to input CSV file.")
    parser.add_argument("--text-col", default="tweet", help="Text column name.")
    parser.add_argument("--label-col", default="class", help="Label column name.")
    parser.add_argument(
        "--label-map",
        choices=["davidson", "hatexplain", "none"],
        default="davidson",
        help="How to map dataset labels to hate_speech/offensive_language/neither.",
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--dev-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models-dir", default="hate_offensive_detection/models")
    parser.add_argument("--figures-dir", default="hate_offensive_detection/reports/figures")
    return parser.parse_args()


def load_dataset(path, text_col, label_col, label_map_name):
    df = pd.read_csv(path)
    missing = [col for col in [text_col, label_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna().copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    if label_map_name != "none":
        mapping = LABEL_MAPS[label_map_name]
        df["label"] = df["label"].map(mapping)

    df = df[df["label"].isin(LABEL_ORDER)].copy()
    if df.empty:
        raise ValueError(
            "No usable rows after label mapping. Check --label-map, --label-col, and labels."
        )

    df["clean_text"] = df["text"].apply(clean_text)
    return df


def split_dataset(df, test_size, dev_size, random_state):
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + dev_size,
        random_state=random_state,
        stratify=df["label"],
    )
    relative_test_size = test_size / (test_size + dev_size)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_df["label"],
    )
    return train_df, dev_df, test_df


def build_models(random_state):
    nb_model = Pipeline(
        [
            (
                "vec",
                CountVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            ("clf", MultinomialNB(alpha=1.0)),
        ]
    )

    logreg_model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=2,
                                max_df=0.95,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "naive_bayes_count": nb_model,
        "logreg_word_char_tfidf": logreg_model,
    }


def evaluate_model(name, model, split_name, texts, labels):
    predictions = model.predict(texts)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    print(f"\n=== {name} on {split_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(labels, predictions, labels=LABEL_ORDER, digits=4))
    return predictions, {"accuracy": accuracy, "macro_f1": macro_f1}


def save_confusion_matrix(labels, predictions, output_path):
    cm = confusion_matrix(labels, predictions, labels=LABEL_ORDER)
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Hate Speech vs Offensive Language Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    figures_dir = Path(args.figures_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data, args.text_col, args.label_col, args.label_map)
    print("\nClass distribution:")
    print(df["label"].value_counts())

    train_df, dev_df, test_df = split_dataset(
        df,
        test_size=args.test_size,
        dev_size=args.dev_size,
        random_state=args.random_state,
    )

    models = build_models(args.random_state)
    dev_scores = {}

    for name, model in models.items():
        model.fit(train_df["clean_text"], train_df["label"])
        _, scores = evaluate_model(
            name,
            model,
            "dev",
            dev_df["clean_text"],
            dev_df["label"],
        )
        dev_scores[name] = scores["macro_f1"]

    best_name = max(dev_scores, key=dev_scores.get)
    best_model = models[best_name]
    print(f"\nBest model by dev macro F1: {best_name}")

    test_predictions, test_scores = evaluate_model(
        best_name,
        best_model,
        "test",
        test_df["clean_text"],
        test_df["label"],
    )

    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nSaved model: {model_path}")

    cm_path = figures_dir / "confusion_matrix.png"
    save_confusion_matrix(test_df["label"], test_predictions, cm_path)
    print(f"Saved confusion matrix: {cm_path}")

    results_path = models_dir / "results.txt"
    with results_path.open("w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Test accuracy: {test_scores['accuracy']:.4f}\n")
        f.write(f"Test macro F1: {test_scores['macro_f1']:.4f}\n")
        f.write("\n")
        f.write(classification_report(test_df["label"], test_predictions, labels=LABEL_ORDER))
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()

