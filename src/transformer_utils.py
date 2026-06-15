"""Shared transformer loading and inference helpers."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import clean_text


DEFAULT_LABEL_ORDER = ["hate_speech", "offensive_language", "neither"]


def _read_json_if_exists(path):
    """Read a JSON file if it exists, otherwise return an empty dictionary."""

    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _label_order_from_config(model):
    """Recover label order from a Hugging Face model config when available."""

    id2label = getattr(model.config, "id2label", None) or {}
    if not id2label:
        return DEFAULT_LABEL_ORDER

    try:
        ordered_keys = sorted(id2label, key=lambda key: int(key))
    except (TypeError, ValueError):
        ordered_keys = sorted(id2label)

    ordered_labels = [id2label[key] for key in ordered_keys]
    return ordered_labels or DEFAULT_LABEL_ORDER


def load_transformer_bundle(model_dir):
    """
    Load the tokenizer, model, and saved metadata for inference.

    The returned bundle is a small dictionary so the Streamlit app can cache
    it and the CLI predictor can reuse the same loading path.
    """

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Transformer model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    metadata = _read_json_if_exists(model_dir / "training_metadata.json")
    metrics = _read_json_if_exists(model_dir / "metrics.json")
    label_order = metadata.get("label_order") or _label_order_from_config(model)
    max_length = int(metadata.get("max_length", 128))

    return {
        "model_dir": str(model_dir),
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "label_order": label_order,
        "max_length": max_length,
        "metadata": metadata,
        "metrics": metrics,
    }


def predict_transformer(bundle, text):
    """
    Predict a label and class probabilities for one input string.

    Returns:
        (label, probability_dict)
    """

    cleaned = clean_text(text)
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    label_order = bundle["label_order"]
    max_length = bundle.get("max_length", 128)

    encoded = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    label_index = int(np.argmax(probabilities))
    label = label_order[label_index]
    prob_dict = {
        label_name: float(probability)
        for label_name, probability in zip(label_order, probabilities)
    }

    return label, prob_dict
