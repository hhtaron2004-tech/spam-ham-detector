import numpy as np
import pandas as pd
from typing import Tuple

def compute_centroids(train_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute class centroids for 'spam' and 'ham' classes from a training BOW matrix.

    Parameters:
    ----------
    train_matrix : pd.DataFrame
        Training dataset containing features and a 'Category' column with labels ('spam' or 'ham').

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        cent_spam : np.ndarray
            Centroid vector for spam class (mean of all spam samples)
        cent_ham : np.ndarray
            Centroid vector for ham class (mean of all ham samples)
    """
    cent_spam = train_matrix[train_matrix["Category"] == "spam"].iloc[:, :-1].mean(axis=0).values
    cent_ham = train_matrix[train_matrix["Category"] == "ham"].iloc[:, :-1].mean(axis=0).values

    return cent_spam, cent_ham


def predict(x: np.ndarray, cent_spam: np.ndarray, cent_ham: np.ndarray) -> str:
    """
    Predict class label ('spam' or 'ham') for a single sample using Euclidean distance to class centroids.

    Parameters:
    ----------
    x : np.ndarray
        Feature vector of a single sample.
    cent_spam : np.ndarray
        Centroid vector for the spam class.
    cent_ham : np.ndarray
        Centroid vector for the ham class.

    Returns:
    -------
    str
        Predicted class label: 'spam' or 'ham'.
    """
    dist_spam = np.linalg.norm(x - cent_spam)
    dist_ham = np.linalg.norm(x - cent_ham)

    return "spam" if dist_spam < dist_ham else "ham"