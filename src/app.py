from pathlib import Path

import joblib
import streamlit as st

from llm_predict import llm_is_configured, predict_llm
from logging_utils import archive_existing_output_txts, make_log_path
from preprocess import clean_text
from transformer_utils import load_transformer_bundle, predict_transformer


LABEL_DISPLAY = {
    "hate_speech": "Hate Speech",
    "offensive_language": "Offensive Language",
    "neither": "Neither / Clean",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_LOGS = archive_existing_output_txts(PROJECT_ROOT)

# Keep the current baseline selection logic: use the hierarchical model when it
# exists, otherwise fall back to the flat logistic-regression bundle.
HIER_PATH = PROJECT_ROOT / "models" / "hierarchical_model.joblib"
FLAT_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
BASELINE_MODEL_PATH = HIER_PATH if HIER_PATH.exists() else FLAT_PATH

DISTILBERT_PATH = PROJECT_ROOT / "models" / "distilbert_hsvol"
HATEBERT_PATH = PROJECT_ROOT / "models" / "hatebert_hsvol"


@st.cache_resource
def load_baseline_model():
    """Load the current sklearn/joblib bundle once per Streamlit session."""

    model_path = HIER_PATH if HIER_PATH.exists() else FLAT_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            "No baseline model found. Train the logistic-regression or "
            "hierarchical model first."
        )
    return joblib.load(model_path)


@st.cache_resource
def load_transformer_model(model_dir):
    """Load and cache a transformer checkpoint for repeated predictions."""

    return load_transformer_bundle(model_dir)


def is_hierarchical_bundle(bundle):
    """Detect the hierarchical bundle saved by src/train_hierarchical.py."""

    return isinstance(bundle, dict) and bundle.get("type") == "hierarchical"


def predict_flat(bundle, text):
    """Run the flat logistic-regression pipeline on one input string."""

    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    threshold = bundle.get("hate_threshold", 0.5) if isinstance(bundle, dict) else 0.5
    cleaned = clean_text(text)

    if not hasattr(model, "predict_proba"):
        label = model.predict([cleaned])[0]
        return label, {}

    probs = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    hate_idx = classes.index("hate_speech")
    label = "hate_speech" if probs[hate_idx] >= threshold else classes[int(probs.argmax())]
    return label, {"Confidence": dict(zip(classes, probs))}


def predict_hierarchical(bundle, text):
    """Run the two-stage hierarchical classifier on one input string."""

    gate_model = bundle["gate_model"]
    severity_model = bundle["severity_model"]
    gate_threshold = bundle.get("gate_threshold", 0.5)
    hate_threshold = bundle.get("hate_threshold", 0.5)
    cleaned = clean_text(text)

    gate_probs = gate_model.predict_proba([cleaned])[0]
    gate_classes = list(gate_model.classes_)
    harmful_idx = gate_classes.index("harmful")
    gate_probs_dict = dict(zip(gate_classes, gate_probs))

    if gate_probs[harmful_idx] < gate_threshold:
        return "neither", {"Stage 1 - Harmful vs Clean": gate_probs_dict}

    severity_probs = severity_model.predict_proba([cleaned])[0]
    severity_classes = list(severity_model.classes_)
    hate_idx = severity_classes.index("hate_speech")
    severity_probs_dict = dict(zip(severity_classes, severity_probs))

    label = "hate_speech" if severity_probs[hate_idx] >= hate_threshold else "offensive_language"
    return label, {
        "Stage 1 - Harmful vs Clean": gate_probs_dict,
        "Stage 2 - Hate Speech vs Offensive Language": severity_probs_dict,
    }


def predict_baseline(bundle, text):
    """Dispatch to the flat or hierarchical baseline bundle."""

    if is_hierarchical_bundle(bundle):
        return predict_hierarchical(bundle, text)
    return predict_flat(bundle, text)


def write_prediction_log(path, text, label, sections):
    """Append one prediction record to the active log file."""

    with Path(path).open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Input text: {text}\n")
        f.write(f"Predicted label: {label}\n")
        for title, probs in (sections or {}).items():
            if not probs:
                continue
            f.write(f"{title}:\n")
            for cls, prob in probs.items():
                f.write(f"  - {cls}: {prob:.4f}\n")


def render_sections(sections):
    """Render each probability section in the Streamlit UI."""

    for title, probs in (sections or {}).items():
        if not probs:
            continue
        st.subheader(title)
        for cls, prob in sorted(probs.items(), key=lambda item: item[1], reverse=True):
            st.progress(float(prob), text=f"{LABEL_DISPLAY.get(cls, cls)}: {prob:.1%}")


def get_model_choice_path(choice):
    """Map the selected model name to the on-disk checkpoint path."""

    if choice == "DistilBERT":
        return DISTILBERT_PATH
    if choice == "HateBERT":
        return HATEBERT_PATH
    return None


# UI -------------------------------------------------------------------------

st.set_page_config(
    page_title="Hate Speech vs Offensive Language Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("Hate Speech vs Offensive Language Detector")
st.write("Paste a sentence or short social-media post to classify it.")

model_choice = st.selectbox(
    "Choose model",
    ["Current Logistic Regression", "DistilBERT", "HateBERT", "LLM/API"],
)

if model_choice == "Current Logistic Regression":
    if not BASELINE_MODEL_PATH.exists():
        st.error(
            "No baseline model found. Train first:\n\n"
            "Hierarchical: `python src/train_hierarchical.py`\n\n"
            "Flat: `python src/train.py --data data/labeled_data.csv "
            "--text-col tweet --label-col class --label-map davidson`"
        )
        st.stop()
    bundle = load_baseline_model()
    model_label = "Hierarchical model" if is_hierarchical_bundle(bundle) else "Flat model"
    model_hint = BASELINE_MODEL_PATH.name
elif model_choice in {"DistilBERT", "HateBERT"}:
    model_dir = get_model_choice_path(model_choice)
    if not model_dir.exists():
        st.error(
            f"No transformer checkpoint found at `{model_dir}`. "
            "Train the transformer first before selecting it here."
        )
        st.stop()
    bundle = load_transformer_model(str(model_dir))
    model_label = f"Transformer: {model_choice}"
    model_hint = model_dir.name
else:
    bundle = None
    model_label = "Optional LLM/API"
    model_hint = "external API"
    if not llm_is_configured():
        st.info(
            "The LLM/API branch is optional and is not configured yet. "
            "Set `LLM_API_KEY` and `LLM_MODEL` and add your client logic in "
            "`src/llm_predict.py` to enable it."
        )

if (
    "prediction_log_path" not in st.session_state
    or st.session_state.get("prediction_log_choice") != model_choice
):
    st.session_state.prediction_log_choice = model_choice
    st.session_state.prediction_log_path = make_log_path(
        PROJECT_ROOT, "prediction_log"
    )
    with st.session_state.prediction_log_path.open("w", encoding="utf-8") as f:
        f.write("Prediction log\n")
        f.write(f"Model choice: {model_choice}\n")
        f.write(f"Model path: {model_hint}\n")
        if ARCHIVED_LOGS:
            f.write("Archived previous output txt files:\n")
            for archived_log in ARCHIVED_LOGS:
                f.write(f"{archived_log}\n")
        f.write("\n")

st.caption(f"Using: {model_label} — `{model_hint}`")
st.caption(f"Prediction log: `{st.session_state.prediction_log_path}`")

user_input = st.text_area("Text", height=180, placeholder="Paste text here...")

if st.button("Check text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            if model_choice == "Current Logistic Regression":
                label, sections = predict_baseline(bundle, user_input)
            elif model_choice in {"DistilBERT", "HateBERT"}:
                label, probs = predict_transformer(bundle, user_input)
                sections = {"Transformer probabilities": probs}
            else:
                label, probs = predict_llm(user_input)
                sections = {"LLM probabilities": probs} if probs else {}
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        display_label = LABEL_DISPLAY.get(label, label)
        write_prediction_log(
            st.session_state.prediction_log_path,
            user_input,
            label,
            sections,
        )

        if label == "hate_speech":
            st.error(display_label)
        elif label == "offensive_language":
            st.warning(display_label)
        else:
            st.success(display_label)

        render_sections(sections)
