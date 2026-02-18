def evaluate(test_matrix, test_df, cent_spam, cent_ham, predict_func):
    """
    Evaluate model accuracy.

    Returns:
        float: Accuracy
    """
    misclassified = 0

    for i, row in enumerate(test_matrix.values):
        prediction = predict_func(row, cent_spam, cent_ham)
        true_label = test_df.iloc[i, 0]

        if prediction != true_label:
            misclassified += 1

    accuracy = 1 - misclassified / len(test_matrix)

    return accuracy, misclassified
