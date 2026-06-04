from datetime import datetime
from pathlib import Path
import joblib
import streamlit as st
from preprocess import clean_text

LABEL_DISPLAY = {
    "hate_speech":        "Hate Speech",
    "offensive_language": "Offensive Language",
    "neither":            "Neither / Clean",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH   = PROJECT_ROOT / "models" / "best_model.joblib"
LOG_PATH     = PROJECT_ROOT / "prediction_log.txt"


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("hate_threshold", 0.5)
    return bundle, 0.5


def predict_text(model, threshold, text):
    cleaned = clean_text(text)
    probabilities = None

    if hasattr(model, "predict_proba"):
        probs    = model.predict_proba([cleaned])[0]
        classes  = list(model.classes_)
        hate_idx = classes.index("hate_speech")
        label = "hate_speech" if probs[hate_idx] >= threshold \
                else classes[int(probs.argmax())]
        probabilities = dict(zip(classes, probs))
    else:
        label = model.predict([cleaned])[0]

    return label, probabilities


def write_prediction_log(text, label, probabilities, threshold):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    display_label = LABEL_DISPLAY.get(label, label)
    lines = [
        "=" * 80,
        f"Timestamp: {timestamp}",
        f"Input text: {text}",
        f"Predicted label: {label}",
        f"Display label: {display_label}",
        f"Hate threshold: {threshold}",
    ]

    if probabilities:
        lines.append("Probabilities:")
        for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {cls}: {prob:.4f}")

    lines.append("")

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hate Speech vs Offensive Language Detector",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Hate Speech vs Offensive Language Detector")
st.write("Paste a sentence or short social-media post to classify it.")

user_input = st.text_area("Text", height=180, placeholder="Paste text here...")

if st.button("Check text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    elif not MODEL_PATH.exists():
        st.error(
            "Model not found. Run: `python src/train.py --data data/labeled_data.csv "
            "--text-col tweet --label-col class --label-map davidson`"
        )
    else:
        model, threshold = load_model()
        label, probabilities = predict_text(model, threshold, user_input)
        write_prediction_log(user_input, label, probabilities, threshold)
        display_label = LABEL_DISPLAY.get(label, label)

        if label == "hate_speech":
            st.error(f"🚨 Result: **{display_label}**")
        elif label == "offensive_language":
            st.warning(f"⚠️ Result: **{display_label}**")
        else:
            st.success(f"✅ Result: **{display_label}**")

        if probabilities:
            st.subheader("Confidence")
            for cls, prob in sorted(probabilities.items(),
                                    key=lambda x: x[1], reverse=True):
                st.progress(float(prob),
                            text=f"{LABEL_DISPLAY.get(cls, cls)}: {prob:.1%}")

        st.caption(f"Saved this prediction to `{LOG_PATH.name}` in the project root.")
