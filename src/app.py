from pathlib import Path

import joblib
import streamlit as st

from preprocess import clean_text


LABEL_DISPLAY = {
    "hate_speech": "Hate Speech",
    "offensive_language": "Offensive Language",
    "neither": "Neither / Clean",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_text(model, text):
    cleaned = clean_text(text)
    label = model.predict([cleaned])[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]

    return label, probabilities


st.set_page_config(
    page_title="Hate Speech vs Offensive Language Detector",
    page_icon="!",
    layout="centered",
)

st.title("Hate Speech vs Offensive Language Detector")
st.write("Paste a sentence or short social-media post to classify it.")

user_input = st.text_area("Text", height=180, placeholder="Paste text here...")

if st.button("Check text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    elif not MODEL_PATH.exists():
        st.error(
            "Model not found. Train it first with "
            "`python src/train.py --data data/labeled_data.csv --text-col tweet "
            "--label-col class --label-map davidson --models-dir models "
            "--figures-dir reports/figures`."
        )
    else:
        model = load_model()
        label, probabilities = predict_text(model, user_input)
        display_label = LABEL_DISPLAY.get(label, label)

        if label == "hate_speech":
            st.error(f"Result: {display_label}")
        elif label == "offensive_language":
            st.warning(f"Result: {display_label}")
        else:
            st.success(f"Result: {display_label}")

        if probabilities is not None:
            st.subheader("Confidence")
            for class_name, probability in zip(model.classes_, probabilities):
                st.write(f"{LABEL_DISPLAY.get(class_name, class_name)}: {probability:.2%}")
        else:
            st.info("This model gives a label but does not provide probabilities.")
