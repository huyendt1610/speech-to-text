import re


def normalize(text):
    """Strip punctuation and lowercase for fair WER comparison."""
    return re.sub(r"[^\w\s]", "", text).lower()
