from pathlib import Path
import joblib
from preprocess import clean_text

LABEL_DISPLAY = {
    "hate_speech":        "Hate Speech",
    "offensive_language": "Offensive Language",
    "neither":            "Neither",
}

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the saved bundle and return (model, threshold)."""
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("hate_threshold", 0.5)
    return bundle, 0.5          # backwards-compatible with old saves


def predict(text: str, model_path=DEFAULT_MODEL_PATH) -> str:
    model, threshold = load_model(model_path)
    cleaned = clean_text(text)

    if hasattr(model, "predict_proba"):
        probs    = model.predict_proba([cleaned])[0]
        classes  = list(model.classes_)
        hate_idx = classes.index("hate_speech")
        label = "hate_speech" if probs[hate_idx] >= threshold \
                else classes[int(probs.argmax())]
    else:
        label = model.predict([cleaned])[0]

    return LABEL_DISPLAY.get(label, label)