import pandas as pd
from sklearn.model_selection import train_test_split
from src import build_vocabulary, create_bow_matrix, compute_centroids, predict, evaluate, to_alpha

def main():

    # Load dataset
    df = pd.read_csv("data/spamhamdata.csv", delimiter="\t", header=None)
    df.columns = ["Category", "Text"]

    # Split dataset
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        shuffle=True,
        random_state=42
    )

    # Build vocabulary
    vocabulary = build_vocabulary(train_df["Text"].values)

    # Create training matrix
    train_matrix = create_bow_matrix(train_df, vocabulary)
    train_matrix["Category"] = train_df["Category"]

    # Compute centroids
    cent_spam, cent_ham = compute_centroids(train_matrix)

    # Create test matrix
    test_matrix = create_bow_matrix(test_df, vocabulary)

    # Evaluate
    accuracy, misclassified = evaluate(
        test_matrix,
        test_df,
        cent_spam,
        cent_ham,
        predict
    )

    print("Test samples:", len(test_matrix))
    print("Misclassified:", misclassified)
    print("Accuracy:", round(accuracy, 4))


if __name__ == "__main__":
    main()
