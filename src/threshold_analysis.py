import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocess import prepare_dataset, preprocess_data


DATA_PATH = "data/raw/KDDTrain+.txt"
RESULTS_DIR = "results"


def calculate_threshold_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tp = ((y_true == 1) & (predictions == 1)).sum()
    tn = ((y_true == 0) & (predictions == 0)).sum()
    fp = ((y_true == 0) & (predictions == 1)).sum()
    fn = ((y_true == 1) & (predictions == 0)).sum()

    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    return {
        "Threshold": threshold,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = prepare_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest for threshold analysis...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    results = []

    for threshold in thresholds:
        metrics = calculate_threshold_metrics(
            y_test,
            probabilities,
            threshold
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\nThreshold Analysis:")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        f"{RESULTS_DIR}/threshold_analysis.csv",
        index=False
    )

    # Plot Recall and FPR against threshold
    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["Threshold"],
        results_df["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        results_df["Threshold"],
        results_df["FPR"],
        marker="o",
        label="False Positive Rate"
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Rate")
    plt.title("Threshold vs Detection and False Positive Rate")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{RESULTS_DIR}/threshold_tradeoff.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nResults saved:")
    print(f"- {RESULTS_DIR}/threshold_analysis.csv")
    print(f"- {RESULTS_DIR}/threshold_tradeoff.png")


if __name__ == "__main__":
    main()