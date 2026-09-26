import os
import joblib

from sklearn.ensemble import RandomForestClassifier

from preprocess import prepare_dataset, preprocess_data


DATA_PATH = "data/raw/KDDTrain+.txt"
MODEL_DIR = "results/model"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading and preprocessing dataset...")

    df = prepare_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        f"{MODEL_DIR}/random_forest.joblib"
    )

    joblib.dump(
        preprocessor,
        f"{MODEL_DIR}/preprocessor.joblib"
    )

    print("\nDeployment model saved:")
    print(f"- {MODEL_DIR}/random_forest.joblib")
    print(f"- {MODEL_DIR}/preprocessor.joblib")


if __name__ == "__main__":
    main()