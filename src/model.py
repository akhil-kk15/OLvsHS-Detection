import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from preprocess import clean_text

LABELS = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "labeled_data.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "svm_model.pkl"


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def train_and_save(
    data_path: Path | str = DEFAULT_DATA_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> None:
    data_path = Path(data_path)
    model_path = Path(model_path)

    df = _read_dataset(data_path)
    df = df.dropna(subset=["tweet", "class"]).copy()
    df["clean_text"] = df["tweet"].apply(clean_text)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["class"]

    model = LinearSVC(class_weight="balanced")
    model.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump((vectorizer, model), f)
    print(f"Model and vectorizer saved successfully to {model_path}.")


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    with model_path.open("rb") as f:
        return pickle.load(f)


def predict(text: str, model_path: Path | str = DEFAULT_MODEL_PATH) -> str:
    vectorizer, model = load_model(model_path)
    cleaned = clean_text(text)
    pred = model.predict(vectorizer.transform([cleaned]))[0]
    return LABELS[pred]
    
    
   
