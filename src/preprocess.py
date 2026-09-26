import pandas as pd
from load_data import load_dataset

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def prepare_dataset(path):
    df = load_dataset(path)

    # Convert multiclass labels into binary classification
    df["target"] = (df["label"] != "normal").astype(int)

    # Remove columns not needed for ML
    df = df.drop(columns=["label", "difficulty"])

    # Remove duplicate records
    df = df.drop_duplicates()

    return df


def preprocess_data(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    categorical_features = [
        "protocol_type",
        "service",
        "flag"
    ]

    numerical_features = [
        column for column in X.columns
        if column not in categorical_features
    ]

    # Split before fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # Encode categorical features and scale numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features
            )
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    df = prepare_dataset("data/raw/KDDTrain+.txt")

    print("Clean dataset shape:", df.shape)

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nMissing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    print("\nProcessed training features:", X_train.shape[1])
    print("Processed testing features:", X_test.shape[1])

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())