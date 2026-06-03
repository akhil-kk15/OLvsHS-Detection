from pathlib import Path

import joblib

from preprocess import clean_text


LABEL_DISPLAY = {
    "hate_speech": "Hate Speech",
    "offensive_language": "Offensive Language",
    "neither": "Neither",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def predict(text: str, model_path: Path | str = DEFAULT_MODEL_PATH) -> str:
    model = load_model(model_path)
    label = model.predict([clean_text(text)])[0]
    return LABEL_DISPLAY.get(label, label)
