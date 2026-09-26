import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from load_data import load_dataset
from preprocess import prepare_dataset, preprocess_data

TRAIN_PATH = "data/raw/KDDTrain+.txt"
TEST_PATH = "data/raw/KDDTest+.txt"
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load original datasets to identify unseen attack types
    train_raw = load_dataset(TRAIN_PATH)
    test_raw = load_dataset(TEST_PATH)

    train_attacks = set(
        train_raw.loc[train_raw["label"] != "normal", "label"].unique()
    )

    test_attacks = set(
        test_raw.loc[test_raw["label"] != "normal", "label"].unique()
    )

    unseen_attacks = sorted(test_attacks - train_attacks)

    print("Attack types seen during training:", len(train_attacks))
    print("Attack types in test set:", len(test_attacks))

    print("\nUnseen attack types:")
    for attack in unseen_attacks:
        print("-", attack)

    # Train Random Forest on training data
    train_df = prepare_dataset(TRAIN_PATH)

    X_train, X_unused, y_train, y_unused, preprocessor = preprocess_data(
        train_df
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # Prepare the complete test dataset using the same preprocessing
    test_df = test_raw.copy()
    test_df["target"] = (test_df["label"] != "normal").astype(int)
    test_labels = test_df["label"].copy()

    test_df = test_df.drop(columns=["label", "difficulty"])

    X_test_raw = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    X_test = preprocessor.transform(X_test_raw)

    predictions = model.predict(X_test)

    # Evaluate only unseen attack samples
    unseen_mask = test_labels.isin(unseen_attacks)

    y_unseen = y_test[unseen_mask]
    predictions_unseen = predictions[unseen_mask]

    print("\nUnseen attack samples:", len(y_unseen))

    if len(y_unseen) > 0:
        print("\nClassification report for unseen attacks:")
        print(
            classification_report(
                y_unseen,
                predictions_unseen,
                target_names=["Normal", "Attack"],
                zero_division=0
            )
        )

        results = pd.DataFrame({
            "Attack Type": test_labels[unseen_mask].values,
            "Actual": y_unseen.values,
            "Predicted": predictions_unseen
        })

        results.to_csv(
            f"{RESULTS_DIR}/unseen_attack_results.csv",
            index=False
        )

    print("\nSaved: results/unseen_attack_results.csv")


if __name__ == "__main__":
    main()