from pathlib import Path
from typing import List, Tuple


def load_imdb_split(split_dir: Path) -> Tuple[List[str], List[int]]:
    """Load text files and binary labels from one IMDB split directory.

    Args:
        split_dir: Path to the "train" or "test" folder inside data/raw/aclImdb.

    Returns:
        A tuple of (texts, labels). Positive reviews are labeled 1 and negative reviews 0.
    """
    texts: List[str] = []
    labels: List[int] = []

    positive_dir = split_dir / "pos"
    negative_dir = split_dir / "neg"

    for text_path in sorted(positive_dir.glob("*.txt")):
        with text_path.open("r", encoding="utf-8", errors="ignore") as handle:
            texts.append(handle.read())
            labels.append(1)

    for text_path in sorted(negative_dir.glob("*.txt")):
        with text_path.open("r", encoding="utf-8", errors="ignore") as handle:
            texts.append(handle.read())
            labels.append(0)

    return texts, labels


def load_imdb_dataset(raw_dir: Path) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load the IMDB dataset from the raw aclImdb folder.

    Args:
        raw_dir: Path to the aclImdb folder containing train/ and test/.

    Returns:
        train_texts, train_labels, test_texts, test_labels.
    """
    train_dir = raw_dir / "train"
    test_dir = raw_dir / "test"

    train_texts, train_labels = load_imdb_split(train_dir)
    test_texts, test_labels = load_imdb_split(test_dir)

    return train_texts, train_labels, test_texts, test_labels
