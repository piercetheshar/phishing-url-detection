"""
Runs the complete ML pipeline:
- Load data
- Feature engineering
- Preprocessing
- Train/test split
- Train baseline models
- Show metrics
"""

from data_loader import load_dataset
from preprocessing import preprocess
from feature_engineering import add_all_features
from modeling import train_test_split_data, train_and_evaluate_models

LABEL_COL = "label"
URL_COL = "URL"
TITLE_COL = "Title"


def main():
    print("📥 Loading dataset...")
    df = load_dataset()

    if LABEL_COL not in df.columns:
        raise KeyError(f"Label column '{LABEL_COL}' not found.")

    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])

    print("🧠 Feature engineering...")
    X = add_all_features(X, url_col=URL_COL, title_col=TITLE_COL)

    print("🧹 Preprocessing...")
    X_processed, _ = preprocess(X)

    print("✂️ Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split_data(X_processed, y)

    print("🤖 Training models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("\n🎯 Summary of models by F1-score:")
    for name, m in sorted(results.items(), key=lambda x: x[1].f1, reverse=True):
        print(f"{name}: F1={m.f1:.4f} Acc={m.accuracy:.4f}")


if __name__ == "__main__":
    main()
