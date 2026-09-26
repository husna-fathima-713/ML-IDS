import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from preprocess import prepare_dataset, preprocess_data


DATA_PATH = "data/raw/KDDTrain+.txt"


def train_models():
    # Load and preprocess data
    df = prepare_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "SVM": SVC(
            class_weight="balanced",
            probability=True,
            random_state=42
        )
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print(f"{name} training accuracy: {train_score:.4f}")
        print(f"{name} testing accuracy:  {test_score:.4f}")


if __name__ == "__main__":
    train_models()