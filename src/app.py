from pathlib import Path
import joblib
import streamlit as st
from logging_utils import archive_existing_output_txts, make_log_path
from preprocess import clean_text

LABEL_DISPLAY = {
    "hate_speech":        "Hate Speech",
    "offensive_language": "Offensive Language",
    "neither":            "Neither / Clean",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_LOGS = archive_existing_output_txts(PROJECT_ROOT)

# Prefer hierarchical model if it exists, fall back to flat
HIER_PATH = PROJECT_ROOT / "models" / "hierarchical_model.joblib"
FLAT_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
MODEL_PATH = HIER_PATH if HIER_PATH.exists() else FLAT_PATH


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_flat(bundle, text):
    model     = bundle["model"] if isinstance(bundle, dict) else bundle
    threshold = bundle.get("hate_threshold", 0.5) if isinstance(bundle, dict) else 0.5
    cleaned   = clean_text(text)

    if not hasattr(model, "predict_proba"):
        return model.predict([cleaned])[0], None, None

    probs    = model.predict_proba([cleaned])[0]
    classes  = list(model.classes_)
    hate_idx = classes.index("hate_speech")
    label    = "hate_speech" if probs[hate_idx] >= threshold \
               else classes[int(probs.argmax())]

    return label, dict(zip(classes, probs)), None


def predict_hierarchical(bundle, text):
    gate_model     = bundle["gate_model"]
    severity_model = bundle["severity_model"]
    gate_threshold = bundle.get("gate_threshold", 0.5)
    hate_threshold = bundle.get("hate_threshold", 0.5)
    cleaned        = clean_text(text)

    gate_probs    = gate_model.predict_proba([cleaned])[0]
    gate_classes  = list(gate_model.classes_)
    harmful_idx   = gate_classes.index("harmful")

    gate_probs_dict = dict(zip(gate_classes, gate_probs))

    if gate_probs[harmful_idx] < gate_threshold:
        return "neither", gate_probs_dict, None

    severity_probs   = severity_model.predict_proba([cleaned])[0]
    severity_classes = list(severity_model.classes_)
    hate_idx         = severity_classes.index("hate_speech")

    label = "hate_speech" if severity_probs[hate_idx] >= hate_threshold \
            else "offensive_language"

    return label, gate_probs_dict, dict(zip(severity_classes, severity_probs))


def write_prediction_log(path, text, label, gate_probs, severity_probs):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Input text: {text}\n")
        f.write(f"Predicted label: {label}\n")
        if gate_probs:
            f.write("Gate / confidence probabilities:\n")
            for cls, prob in gate_probs.items():
                f.write(f"  - {cls}: {prob:.4f}\n")
        if severity_probs:
            f.write("Severity probabilities:\n")
            for cls, prob in severity_probs.items():
                f.write(f"  - {cls}: {prob:.4f}\n")


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hate Speech vs Offensive Language Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Hate Speech vs Offensive Language Detector")
st.write("Paste a sentence or short social-media post to classify it.")

if not MODEL_PATH.exists():
    st.error(
        "No model found. Train first:\n\n"
        "Hierarchical: `python src/train_hierarchical.py`\n\n"
        "Flat: `python src/train.py --data data/labeled_data.csv "
        "--text-col tweet --label-col class --label-map davidson`"
    )
    st.stop()

bundle       = load_model()
is_hier      = isinstance(bundle, dict) and bundle.get("type") == "hierarchical"
model_label  = "Hierarchical model" if is_hier else "Flat model"

if "prediction_log_path" not in st.session_state:
    st.session_state.prediction_log_path = make_log_path(
        PROJECT_ROOT, "prediction_log"
    )
    with st.session_state.prediction_log_path.open("w", encoding="utf-8") as f:
        f.write("Prediction log\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        if ARCHIVED_LOGS:
            f.write("Archived previous output txt files:\n")
            for archived_log in ARCHIVED_LOGS:
                f.write(f"{archived_log}\n")
        f.write("\n")

st.caption(f"Using: {model_label} — `{MODEL_PATH.name}`")
st.caption(f"Prediction log: `{st.session_state.prediction_log_path}`")

user_input = st.text_area("Text", height=180, placeholder="Paste text here...")

if st.button("Check text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        if is_hier:
            label, gate_probs, severity_probs = predict_hierarchical(bundle, user_input)
        else:
            label, gate_probs, severity_probs = predict_flat(bundle, user_input)

        display_label = LABEL_DISPLAY.get(label, label)
        write_prediction_log(
            st.session_state.prediction_log_path,
            user_input,
            label,
            gate_probs,
            severity_probs,
        )

        # ── Result banner ──
        if label == "hate_speech":
            st.error(f"🚨 **{display_label}**")
        elif label == "offensive_language":
            st.warning(f"⚠️ **{display_label}**")
        else:
            st.success(f"✅ **{display_label}**")

        # ── Probability breakdown ──
        if gate_probs:
            if is_hier:
                st.subheader("Stage 1 — Harmful vs Clean")
                for cls in ["harmful", "neither"]:
                    prob = gate_probs.get(cls, 0.0)
                    st.progress(float(prob), text=f"{cls.capitalize()}: {prob:.1%}")

                if severity_probs:
                    st.subheader("Stage 2 — Hate Speech vs Offensive")
                    for cls in ["hate_speech", "offensive_language"]:
                        prob = severity_probs.get(cls, 0.0)
                        st.progress(
                            float(prob),
                            text=f"{LABEL_DISPLAY.get(cls, cls)}: {prob:.1%}"
                        )
                else:
                    st.info("Text was classified as clean at Stage 1 — "
                            "severity stage was not reached.")
            else:
                st.subheader("Confidence")
                for cls, prob in sorted(gate_probs.items(),
                                        key=lambda x: x[1], reverse=True):
                    st.progress(
                        float(prob),
                        text=f"{LABEL_DISPLAY.get(cls, cls)}: {prob:.1%}"
                    )
