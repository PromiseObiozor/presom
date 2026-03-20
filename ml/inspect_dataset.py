import pandas as pd
from pathlib import Path

DATA_PATH = Path("ml/data/emails.csv")

def main():
    print("Script started...")

    if not DATA_PATH.exists():
        print(f"Dataset not found at: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== DATASET SHAPE ===")
    print(df.shape)

    print("\n=== NULL VALUES ===")
    print(df.isnull().sum())

    print("\n=== LABEL DISTRIBUTION ===")
    possible_label_cols = ["label", "Label", "class", "Class", "target"]
    label_col = None

    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col:
        print(df[label_col].value_counts(dropna=False))
    else:
        print("No recognised label column found.")

if __name__ == "__main__":
    main()