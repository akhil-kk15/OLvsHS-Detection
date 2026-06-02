import argparse
import joblib
from preprocess import clean_text


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one text label.")
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model)
    cleaned = clean_text(args.text)
    prediction = model.predict([cleaned])[0]

    print(f"Prediction: {prediction}")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        for label, probability in zip(model.classes_, probabilities):
            print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()
