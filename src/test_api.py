import requests
import pandas as pd

from load_data import load_dataset


TEST_PATH = "data/raw/KDDTest+.txt"


def main():
    df = load_dataset(TEST_PATH)

    # Select one real attack record
    attack_row = df[df["label"] != "normal"].iloc[0].copy()

    # Remove fields that are not model inputs
    attack_row = attack_row.drop(
        labels=["label", "difficulty"]
    )

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=attack_row.to_dict()
    )

    print("API response:")
    print(response.json())


if __name__ == "__main__":
    main()