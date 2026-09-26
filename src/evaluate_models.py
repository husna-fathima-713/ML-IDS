import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    ConfusionMatrixDisplay
)

from preprocess import prepare_dataset, preprocess_data


DATA_PATH = "data/raw/KDDTrain+.txt"
RESULTS_DIR = "results"


def calculate_metrics(y_true, y_pred, y_scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    pr_auc = average_precision_score(y_true, y_scores)

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "PR-AUC": pr_auc
    }


def evaluate_models():
    os.makedirs(RESULTS_DIR, exist_ok=True)

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

    all_results = []
    confusion_matrices = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities
        )

        metrics["Model"] = name
        all_results.append(metrics)

        cm = confusion_matrix(y_test, predictions)
        confusion_matrices.append((name, cm))

        print("\nConfusion Matrix:")
        print(
            f"TN: {metrics['TN']}, "
            f"FP: {metrics['FP']}, "
            f"FN: {metrics['FN']}, "
            f"TP: {metrics['TP']}"
        )

        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        print(f"F1 Score:  {metrics['F1']:.4f}")
        print(f"FPR:       {metrics['FPR']:.4f}")
        print(f"FNR:       {metrics['FNR']:.4f}")
        print(f"PR-AUC:    {metrics['PR-AUC']:.4f}")

    # Save metrics
    results_df = pd.DataFrame(all_results)

    columns = [
        "Model",
        "TN",
        "FP",
        "FN",
        "TP",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "FNR",
        "PR-AUC"
    ]

    results_df = results_df[columns]
    results_df.to_csv(
        f"{RESULTS_DIR}/model_metrics.csv",
        index=False
    )

    # Save confusion matrix figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, cm) in zip(axes, confusion_matrices):
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Attack"]
        )

        display.plot(
            ax=ax,
            values_format="d",
            colorbar=False
        )

        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_DIR}/confusion_matrices.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nResults saved:")
    print(f"- {RESULTS_DIR}/model_metrics.csv")
    print(f"- {RESULTS_DIR}/confusion_matrices.png")


if __name__ == "__main__":
    evaluate_models()