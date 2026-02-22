from typing import Callable, Tuple
import numpy as np
import pandas as pd


def evaluate(
        test_matrix: pd.DataFrame,
        test_df: pd.DataFrame,
        cent_spam: np.ndarray,
        cent_ham: np.ndarray,
        predict_func: Callable[[np.ndarray, np.ndarray, np.ndarray], str]
) -> Tuple[float, int]:
    """
    Evaluate the accuracy of a classification model on test data.

    Parameters:
    ----------
    test_matrix : pd.DataFrame
        Feature matrix of the test dataset (samples × features).

    test_df : pd.DataFrame
        Original test dataset containing true labels in the first column.

    cent_spam : list
        Centroid vector or reference for the 'spam' class.

    cent_ham : list
        Centroid vector or reference for the 'ham' class.

    predict_func : Callable[[list, list, list], int]
        Function that takes a feature vector, cent_spam, cent_ham and returns
        a predicted label (e.g., 0 or 1).

    Returns:
    -------
    Tuple[float, int]
        accuracy : float
            Fraction of correctly classified samples (between 0 and 1).
        misclassified : int
            Number of misclassified samples.

    """
    misclassified = 0

    for i, row in enumerate(test_matrix.values):

        prediction = predict_func(row, cent_spam, cent_ham)

        true_label = test_df.iloc[i, 0]

        if prediction != true_label:
            misclassified += 1


    accuracy = 1 - misclassified / len(test_matrix)

    return accuracy, misclassified