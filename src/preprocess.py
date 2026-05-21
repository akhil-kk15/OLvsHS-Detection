import html
import re


def clean_text(text: str) -> str:
    """Normalize social-media style text without removing useful abuse signals."""
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

