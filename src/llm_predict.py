# futureImplementation: This file is a placeholder for an API-backed LLM classifier. It checks if the necessary API settings are present and raises an error if they are not configured. The `predict_llm` function is intended to be implemented later with a hosted model, and it should return a label and a probability dictionary consistent with other predictors.

import os


def llm_is_configured():
    """Return True only when the optional API settings are present."""

    return bool(os.getenv("LLM_API_KEY") and os.getenv("LLM_MODEL"))


def predict_llm(text):
    """
    Placeholder for an API-backed LLM classifier.

    If you later wire in a hosted model, keep the return shape consistent with
    the other predictors:

        (label, probability_dict)
    """

    raise RuntimeError(
        "LLM/API prediction is not configured. Set LLM_API_KEY and LLM_MODEL "
        "and add your client implementation in src/llm_predict.py."
    )
