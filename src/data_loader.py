"""
Loads the phishing.csv file from /data.
"""

import pandas as pd
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "phishing.csv"


def load_dataset(path: str = None):
    """
    Loads the phishing URL dataset into a pandas DataFrame.
    """
    csv_path = Path(path) if path else DEFAULT_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
