import numpy as np
import pandas as pd
from typing import Iterable
from .preprocessing import to_alpha


def build_vocabulary(texts: Iterable[str]) -> np.ndarray:
    """
    Build a vocabulary of unique words from a collection of text samples.

    Parameters:
    ----------
    texts : Iterable[str]
        Collection of text samples (e.g., a list or pandas Series of strings).

    Returns:
    -------
    np.ndarray
        Array of unique words in the corpus. Words with length < 2 are ignored.

    Notes:
    -----
    - Uses the `to_alpha` function to normalize text (e.g., remove punctuation, convert to lowercase).
    - Each word appears only once in the returned array.
    """
    words = []

    for text in texts:
        for word in to_alpha(text).split():
            if len(word) >= 2:
                words.append(word)

    return np.unique(words)


def create_bow_matrix(df: pd.DataFrame, vocabulary: np.ndarray) -> pd.DataFrame:
    """
    Create a Bag-of-Words (BOW) matrix from a DataFrame of text samples.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing a 'Text' column with text samples.

    vocabulary : np.ndarray
        Array of unique words (from `build_vocabulary`).

    Returns:
    -------
    pd.DataFrame
        Bag-of-Words matrix:
        - Rows correspond to samples in `df`.
        - Columns correspond to words in `vocabulary`.
        - Values are word counts in each sample.

    Notes:
    -----
    - Uses the `to_alpha` function to normalize text.
    - Words not in the vocabulary are ignored.
    """
    n = len(df)
    m = len(vocabulary)

    # Initialize zero matrix
    matrix = pd.DataFrame(
        np.zeros((n, m), dtype=int),
        columns=vocabulary,
        index=df.index
    )

    # Fill matrix with word counts
    for idx, text in zip(df.index, df["Text"].values):
        for word in to_alpha(text).split():
            if word in vocabulary:
                matrix.loc[idx, word] += 1

    return matrix