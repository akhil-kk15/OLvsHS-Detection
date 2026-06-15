"""Command line prediction entry point for transformer checkpoints."""

import argparse
from pathlib import Path

from preprocess import clean_text
from transformer_utils import load_transformer_bundle, predict_transformer


LABEL_DISPLAY = {
    "hate_speech": "Hate Speech",
    "offensive_language": "Offensive Language",
    "neither": "Neither / Clean",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one text with a transformer.")
    parser.add_argument("--model-dir", default="models/distilbert_hsvol")
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    bundle = load_transformer_bundle(model_dir)
    cleaned = clean_text(args.text)
    label, probabilities = predict_transformer(bundle, args.text)

    print(f"Model: {model_dir}")
    print(f"Input:   {args.text}")
    print(f"Cleaned: {cleaned}\n")
    print(f"Prediction: {LABEL_DISPLAY.get(label, label)}")
    for cls, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"  {LABEL_DISPLAY.get(cls, cls)}: {prob:.4f}")


if __name__ == "__main__":
    main()
