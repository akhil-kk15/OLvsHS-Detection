from pathlib import Path
import joblib
from preprocess import clean_text

LABEL_DISPLAY = {
    "hate_speech":        "Hate Speech",
    "offensive_language": "Offensive Language",
    "neither":            "Neither",
}

PROJECT_ROOT       = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
HIER_MODEL_PATH    = PROJECT_ROOT / "models" / "hierarchical_model.joblib"


def _patch_legacy_logreg_attrs(estimator):
    """
    Patch older scikit-learn LogisticRegression pickles that are missing
    attributes expected by newer runtime versions.

    Some saved models from older sklearn releases do not carry `multi_class`.
    Newer runtimes touch that attribute during prediction, so we inject the
    default value when it is missing.
    """

    if estimator is None:
        return estimator

    if estimator.__class__.__name__ == "LogisticRegression" and not hasattr(
        estimator, "multi_class"
    ):
        estimator.multi_class = "auto"

    named_steps = getattr(estimator, "named_steps", None)
    if named_steps:
        for step in named_steps.values():
            _patch_legacy_logreg_attrs(step)
        return estimator

    steps = getattr(estimator, "steps", None)
    if steps:
        for _, step in steps:
            _patch_legacy_logreg_attrs(step)

    return estimator


def normalize_loaded_bundle(bundle):
    """Patch known compatibility issues on a loaded baseline bundle."""

    if isinstance(bundle, dict):
        for key in ("model", "gate_model", "severity_model"):
            if key in bundle:
                bundle[key] = _patch_legacy_logreg_attrs(bundle[key])
        return bundle

    return _patch_legacy_logreg_attrs(bundle)


def _predict_flat(bundle, cleaned):
    model     = bundle["model"] if isinstance(bundle, dict) else bundle
    threshold = bundle.get("hate_threshold", 0.5) if isinstance(bundle, dict) else 0.5

    if hasattr(model, "predict_proba"):
        probs    = model.predict_proba([cleaned])[0]
        classes  = list(model.classes_)
        hate_idx = classes.index("hate_speech")
        return "hate_speech" if probs[hate_idx] >= threshold \
               else classes[int(probs.argmax())]
    return model.predict([cleaned])[0]


def _predict_hierarchical(bundle, cleaned):
    gate_model     = bundle["gate_model"]
    severity_model = bundle["severity_model"]
    gate_threshold = bundle.get("gate_threshold", 0.5)
    hate_threshold = bundle.get("hate_threshold", 0.5)

    gate_probs    = gate_model.predict_proba([cleaned])[0]
    gate_classes  = list(gate_model.classes_)
    harmful_idx   = gate_classes.index("harmful")

    if gate_probs[harmful_idx] < gate_threshold:
        return "neither"

    severity_probs   = severity_model.predict_proba([cleaned])[0]
    severity_classes = list(severity_model.classes_)
    hate_idx         = severity_classes.index("hate_speech")

    return "hate_speech" if severity_probs[hate_idx] >= hate_threshold \
           else "offensive_language"


def predict(text: str, model_path: Path | str = DEFAULT_MODEL_PATH) -> str:
    bundle  = joblib.load(model_path)
    cleaned = clean_text(text)

    if isinstance(bundle, dict) and bundle.get("type") == "hierarchical":
        label = _predict_hierarchical(bundle, cleaned)
    else:
        label = _predict_flat(bundle, cleaned)

    return LABEL_DISPLAY.get(label, label)
