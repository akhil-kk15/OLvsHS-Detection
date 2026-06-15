"""Optional LLM/API prediction hook.

This project does not depend on an external API key by default. The module is
kept as a small, explicit placeholder so the Streamlit app can show the branch
without breaking the local baseline or transformer workflow.
"""

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
