import numpy as np
import pandas as pd
from preprocessing import to_alpha


def build_vocabulary(texts):
    """
    Build vocabulary from training texts.

    Args:
        texts (iterable): Collection of text samples.

    Returns:
        np.ndarray: Unique words array.
    """
    words = []

    for text in texts:
        for word in to_alpha(text).split():
            if len(word) >= 2:
                words.append(word)

    return np.unique(words)


def create_bow_matrix(df, vocabulary):
    """
    Create Bag-of-Words matrix.

    Args:
        df (DataFrame): DataFrame with 'Text' column.
        vocabulary (array): Unique words list.

    Returns:
        DataFrame: Bag-of-Words matrix.
    """
    n = len(df)
    m = len(vocabulary)

    matrix = pd.DataFrame(
        np.zeros((n, m)),
        columns=vocabulary,
        index=df.index
    )

    for idx, text in zip(df.index, df["Text"].values):
        for word in to_alpha(text).split():
            if word in vocabulary:
                matrix.loc[idx, word] += 1

    return matrix
