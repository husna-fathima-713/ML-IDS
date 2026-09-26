import pandas as pd

INPUT_FILE = "results/model_metrics.csv"
OUTPUT_FILE = "results/final_model_comparison.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    columns = [
        "Model",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "FNR",
        "PR-AUC"
    ]

    comparison = df[columns].copy()

    for column in [
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "FNR",
        "PR-AUC"
    ]:
        comparison[column] = comparison[column].round(4)

    print("\nFinal Model Comparison:")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()