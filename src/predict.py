import argparse
import joblib
from model import normalize_loaded_bundle
from preprocess import clean_text

LABEL_DISPLAY = {
    "hate_speech":        "Hate Speech",
    "offensive_language": "Offensive Language",
    "neither":            "Neither",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one text label.")
    parser.add_argument("--model", default="models/hierarchical_model.joblib")
    #parser("--model", default="models/hierarchical_model.joblib")
    parser.add_argument("--text",  required=True)
    return parser.parse_args()


def predict_flat(bundle, cleaned):
    model     = bundle["model"] if isinstance(bundle, dict) else bundle
    threshold = bundle.get("hate_threshold", 0.5) if isinstance(bundle, dict) else 0.5

    print(f"Hate threshold: {threshold:.2f}")

    if hasattr(model, "predict_proba"):
        probs    = model.predict_proba([cleaned])[0]
        classes  = list(model.classes_)
        hate_idx = classes.index("hate_speech")
        label    = "hate_speech" if probs[hate_idx] >= threshold \
                   else classes[int(probs.argmax())]
        for cls, prob in zip(classes, probs):
            print(f"  {LABEL_DISPLAY.get(cls, cls)}: {prob:.4f}")
        return label
    return model.predict([cleaned])[0]

#the gate threshold was .5 before. 
def predict_hierarchical(bundle, cleaned):
    gate_model     = bundle["gate_model"]
    severity_model = bundle["severity_model"]
    gate_threshold = bundle.get("gate_threshold", 0.3)
    hate_threshold = bundle.get("hate_threshold", 0.3)

    print(f"Gate threshold:     {gate_threshold:.2f}")
    print(f"Severity threshold: {hate_threshold:.2f}")

    gate_probs   = gate_model.predict_proba([cleaned])[0]
    gate_classes = list(gate_model.classes_)
    harmful_idx  = gate_classes.index("harmful")

    print("\nStage 1 — Harmful vs Clean:")
    for cls, prob in zip(gate_classes, gate_probs):
        print(f"  {cls}: {prob:.4f}")

    if gate_probs[harmful_idx] < gate_threshold:
        return "neither"

    severity_probs   = severity_model.predict_proba([cleaned])[0]
    severity_classes = list(severity_model.classes_)
    hate_idx         = severity_classes.index("hate_speech")

    print("\nStage 2 — Hate vs Offensive:")
    for cls, prob in zip(severity_classes, severity_probs):
        print(f"  {LABEL_DISPLAY.get(cls, cls)}: {prob:.4f}")

    return "hate_speech" if severity_probs[hate_idx] >= hate_threshold \
           else "offensive_language"


def main():
    args    = parse_args()
    bundle  = normalize_loaded_bundle(joblib.load(args.model))
    cleaned = clean_text(args.text)

    print(f"Input:   {args.text}")
    print(f"Cleaned: {cleaned}\n")

    if isinstance(bundle, dict) and bundle.get("type") == "hierarchical":
        label = predict_hierarchical(bundle, cleaned)
    else:
        label = predict_flat(bundle, cleaned)

    print(f"\nPrediction: {LABEL_DISPLAY.get(label, label)}")


if __name__ == "__main__":
    main()
