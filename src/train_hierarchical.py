import argparse
#from email import parser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from lexiconfeatures import LexiconFeatures
from preprocess import clean_text
from train import LABEL_MAPS,LABEL_ORDER,split_dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a hierarchical model using both Davidson and  OLID datasets."

    )
    parser.add_argument("--data-dir", default= "data")
    parser.add_argument("--models-dir", default= "models")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--dev-size", type=float, default=0.15)
    parser.add_argument("--output-name", default= "hierarchical_model.joblib")
    return parser.parse_args()


def build_features():
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,max_df=0.95, sublinear_tf=True)),
        ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ("lexicon", LexiconFeatures()),
    ])

def build_gate_model(random_state):
    return Pipeline([
        ("features", build_features()),
        ("classifier", LogisticRegression(random_state=random_state, max_iter=1000)),
    ])

def build_severity_model(random_state, hate_count):
    target_hate = max(1500, hate_count)
    return ImbPipeline([
        ("features", build_features()),
        ("oversample", RandomOverSampler(
            sampling_strategy={"hate_speech": target_hate},
            random_state=random_state,
        )),
        ("clf", LogisticRegression(max_iter=3000, random_state=random_state)),
    ])

def load_davidson(data_dir):
    df = pd.read_csv(data_dir / "labeled_data.csv")
    df = df[["tweet", "class"]].dropna().rename(
        columns={"tweet": "text", "class": "label"}
    )
    df["label"] = df["label"].map(LABEL_MAPS["davidson"])
    df = df[df["label"].isin(LABEL_ORDER)].copy()
    df["source"] = "davidson"
    df["clean_text"] = df["text"].apply(clean_text)
    return df

def binary_from_three_class(df):
    out = df[["text","clean_text","source"]].copy()
    out["binary_label"] =np.where(df["label"] == "neither", "neither", "harmful")
    return out

def load_olid_training(data_dir):
    df = pd.read_csv(data_dir / "olid-training-v1.0.tsv", sep="\t")
    out = pd.DataFrame({
        "text": df["tweet"],
        "binary_label": df["subtask_a"].map({"OFF": "harmful", "NOT": "neither"}),
        "source": "olid_2019_train",
    })
    out["clean_text"] = out["text"].apply(clean_text)
    return out.dropna(subset=["binary_label"])

def load_level_a(data_dir, tweets_name, labels_name, source):
    tweets = pd.read_csv(data_dir / tweets_name, sep="\t")
    labels = pd.read_csv(data_dir / labels_name, header=None, names=["id", "label"])
    df = tweets.merge(labels, on="id")
    out = pd.DataFrame({
        "text": df["tweet"],
        "binary_label": df["label"].map({"OFF": "harmful", "NOT": "neither"}),
        "source": source,
    })
    out["clean_text"] = out["text"].apply(clean_text)
    return out.dropna(subset=["binary_label"])

def tune_thresholds(gate_model, severity_model, dev_df):
    gate_probs = gate_model.predict_proba(dev_df["clean_text"])
    gate_classes = list(gate_model.classes_)
    harmful_idx = gate_classes.index("harmful")

    severity_probs = severity_model.predict_proba(dev_df["clean_text"])
    severity_classes = list(severity_model.classes_)
    hate_idx = severity_classes.index("hate_speech")

    best = {"macro_f1": -1.0, "gate_threshold": 0.5, "hate_threshold": 0.5}
    for gate_t in np.arange(0.35, 0.76, 0.05):
        for hate_t in np.arange(0.35, 0.76, 0.05):
            preds = []
            for gate_row, severity_row in zip(gate_probs, severity_probs):
                if gate_row[harmful_idx] < gate_t:
                    preds.append("neither")
                elif severity_row[hate_idx] >= hate_t:
                    preds.append("hate_speech")
                else:
                    preds.append("offensive_language")

            macro_f1 = f1_score(
                dev_df["label"], preds, average="macro", zero_division=0
            )
            hate_f1 = f1_score(
                dev_df["label"], preds, labels=["hate_speech"],
                average="macro", zero_division=0,
            )
            if macro_f1 > best["macro_f1"]:
                best = {
                    "macro_f1": macro_f1,
                    "hate_f1": hate_f1,
                    "gate_threshold": float(gate_t),
                    "hate_threshold": float(hate_t),
                }
    return best

def predict_hierarchical(bundle, texts):
    gate_model = bundle["gate_model"]
    severity_model = bundle["severity_model"]
    gate_threshold = bundle["gate_threshold"]
    hate_threshold = bundle["hate_threshold"]

    gate_probs = gate_model.predict_proba(texts)
    gate_classes = list(gate_model.classes_)
    harmful_idx = gate_classes.index("harmful")

    severity_probs = severity_model.predict_proba(texts)
    severity_classes = list(severity_model.classes_)
    hate_idx = severity_classes.index("hate_speech")

    preds = []
    for gate_row, severity_row in zip(gate_probs, severity_probs):
        if gate_row[harmful_idx] < gate_threshold:
            preds.append("neither")
        elif severity_row[hate_idx] >= hate_threshold:
            preds.append("hate_speech")
        else:
            preds.append("offensive_language")
    return preds

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    davidson_df = load_davidson(data_dir)
    train_df, dev_df, test_df = split_dataset(
        davidson_df, args.test_size, args.dev_size, args.random_state
    )

    gate_train = pd.concat([
        binary_from_three_class(train_df),
        load_olid_training(data_dir),
        load_level_a(data_dir, "testset-levela.tsv", "labels-levela.csv",
                     "olid_2019_test_a"),
        load_level_a(data_dir, "test_a_tweets_all.tsv", "test_a_labels_all.csv",
                     "offenseval_2020_test_a"),
    ], ignore_index=True)
    gate_train = gate_train.drop_duplicates(subset=["clean_text", "binary_label"])

    severity_train = train_df[
        train_df["label"].isin(["hate_speech", "offensive_language"])
    ].copy()

    gate_model = build_gate_model(args.random_state)
    severity_model = build_severity_model(
        args.random_state,
        hate_count=int((severity_train["label"] == "hate_speech").sum()),
    )

    gate_model.fit(gate_train["clean_text"], gate_train["binary_label"])
    severity_model.fit(severity_train["clean_text"], severity_train["label"])

    thresholds = tune_thresholds(gate_model, severity_model, dev_df)
    bundle = {
        "type": "hierarchical",
        "gate_model": gate_model,
        "severity_model": severity_model,
        "gate_threshold": thresholds["gate_threshold"],
        "hate_threshold": thresholds["hate_threshold"],
        "label_order": LABEL_ORDER,
    }

    test_preds = predict_hierarchical(bundle, test_df["clean_text"])
    acc = accuracy_score(test_df["label"], test_preds)
    macro_f1 = f1_score(test_df["label"], test_preds, average="macro", zero_division=0)
    report = classification_report(
        test_df["label"], test_preds, labels=LABEL_ORDER, digits=4, zero_division=0
    )

    print(f"Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")
    print(report)

    joblib.dump(bundle, models_dir / args.output_name)
    with (models_dir / "hierarchical_results.txt").open("w", encoding="utf-8") as f:
        f.write("Model: hierarchical\n")
        f.write(f"Gate threshold: {thresholds['gate_threshold']:.2f}\n")
        f.write(f"Hate threshold: {thresholds['hate_threshold']:.2f}\n")
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Test macro F1: {macro_f1:.4f}\n\n")
        f.write(report)


if __name__ == "__main__":
    main()