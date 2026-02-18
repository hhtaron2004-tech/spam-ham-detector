import numpy as np


def compute_centroids(train_matrix):
    """
    Compute class centroids for spam and ham.

    Args:
        train_matrix (DataFrame): BOW matrix with 'Category' column.

    Returns:
        tuple: (cent_spam, cent_ham)
    """
    cent_spam = train_matrix[train_matrix["Category"] == "spam"] \
        .iloc[:, :-1].mean(axis=0).values

    cent_ham = train_matrix[train_matrix["Category"] == "ham"] \
        .iloc[:, :-1].mean(axis=0).values

    return cent_spam, cent_ham


def predict(x, cent_spam, cent_ham):
    """
    Predict class using Euclidean distance.
    """
    dist_spam = np.linalg.norm(x - cent_spam)
    dist_ham = np.linalg.norm(x - cent_ham)

    return "spam" if dist_spam < dist_ham else "ham"
