"""
Model training + evaluation for Phase 2.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split


@dataclass
class ModelResults:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray
    report: str


def train_test_split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


def get_baseline_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "SVM_rbf": SVC(kernel="rbf", probability=True)
    }


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = float("nan")

    return ModelResults(
        name=name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc,
        confusion=confusion_matrix(y_test, y_pred),
        report=classification_report(y_test, y_pred),
    )


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = get_baseline_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)

        metrics = evaluate_model(name, model, X_test, y_test)
        print(metrics.report)

        results[name] = metrics

    return results
