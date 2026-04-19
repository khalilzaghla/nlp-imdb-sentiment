import re
from typing import Iterable, List, Optional


CLEANING_PATTERN = re.compile(r"[^a-z0-9\s.,!?'-]", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)
HTML_BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Lowercase and perform basic HTML and punctuation-preserving cleaning."""
    text = text.lower()
    text = HTML_BREAK_PATTERN.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = CLEANING_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def preprocess_texts(texts: Iterable[str], max_length: Optional[int] = None) -> List[str]:
    """Preprocess a list of text strings for vectorization.

    This pipeline is shared by both baseline models.

    Args:
        texts: Raw document strings.
        max_length: Optional maximum number of tokens to keep per document.

    Returns:
        A list of cleaned text strings.
    """
    processed_texts: List[str] = []

    for document in texts:
        cleaned = clean_text(document)
        if max_length is not None:
            tokens = cleaned.split()
            cleaned = " ".join(tokens[:max_length])
        processed_texts.append(cleaned)

    return processed_texts
