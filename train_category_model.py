#!/usr/bin/env python3
"""
Train a classifier to predict a target column (default 'category') from a CSV.
- Handles numeric + categorical features with impute/one-hot/scale
- Stratified train/test split
- Choice of model (rf, histgb, logreg, or auto CV selection)
- Saves model, metrics, confusion matrix, and test predictions
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib


def detect_target_column(df: pd.DataFrame, requested: str) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    if requested.lower() not in lower_map:
        raise ValueError(f"Target column matching '{requested}' not found. Available: {list(df.columns)}")
    return lower_map[requested.lower()]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    import numpy as np
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return preprocess, numeric_features, categorical_features


def get_model(name: str):
    name = name.lower()
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    if name == "histgb":
        return HistGradientBoostingClassifier(random_state=42)
    if name == "logreg":
        return LogisticRegression(max_iter=2000, solver="lbfgs")
    raise ValueError(f"Unknown model '{name}'. Choose from: rf, histgb, logreg.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default = '/content/AgariX/Data/s9_wells_rgb_labeled_clinical.csv', help="Path to input CSV")
    p.add_argument("--target", default="category", help="Target column name (case-insensitive). Default: category")
    p.add_argument("--test_size", type=float, default=0.2, help="Test size fraction. Default: 0.2")
    p.add_argument("--model", default="rf", choices=["rf", "histgb", "logreg", "auto"],
                   help="Model to use. 'auto' tries rf/histgb/logreg via CV and picks the best.")
    p.add_argument("--cv", type=int, default=0, help="If >1 and model='auto', do CV (folds) on training set.")
    p.add_argument("--out", default="./outputs", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load
    df = pd.read_csv(args.csv)
    # Drop empty columns
    df = df.dropna(axis=1, how="all")

    target_col = detect_target_column(df, args.target)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    preprocess, num_cols, cat_cols = build_preprocessor(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Choose model
    chosen_name = None
    if args.model == "auto" and args.cv and args.cv > 1:
        candidates = {
            "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=args.seed),
            "histgb": HistGradientBoostingClassifier(random_state=args.seed),
            "logreg": LogisticRegression(max_iter=2000, solver="lbfgs"),
        }
        best_score = -np.inf
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        for name, clf in candidates.items():
            pipe = Pipeline([("prep", preprocess), ("clf", clf)])
            scores = cross_val_score(pipe, X_train, y_train, scoring="f1_macro", cv=skf, n_jobs=-1)
            if scores.mean() > best_score:
                best_score = scores.mean()
                chosen_name = name
        model = candidates[chosen_name]
    else:
        chosen_name = args.model
        model = get_model(chosen_name)

    # Fit
    pipe = Pipeline([("prep", preprocess), ("clf", model)])
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report_text = classification_report(y_test, y_pred)

    # Save artifacts
    model_path = os.path.join(args.out, f"model_{chosen_name}.joblib")
    joblib.dump(pipe, model_path)

    metrics = {
        "csv": os.path.abspath(args.csv),
        "target_column": target_col,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "numeric_features_count": len(num_cols),
        "categorical_features_count": len(cat_cols),
        "model": chosen_name,
        "test_accuracy": acc,
        "test_f1_macro": f1_macro,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.out, "classification_report.txt"), "w") as f:
        f.write(report_text)

    # Confusion matrix
    labels_sorted = sorted(y.unique())
    fig = plt.figure()
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels_sorted)), labels_sorted, rotation=45, ha="right")
    plt.yticks(range(len(labels_sorted)), labels_sorted)
    plt.tight_layout()
    cm_path = os.path.join(args.out, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)

    # Predictions
    preds = pd.DataFrame({"index": X_test.index, "true": y_test.values, "pred": y_pred})
    preds.to_csv(os.path.join(args.out, "test_predictions.csv"), index=False)

    print(json.dumps({
        "saved": {
            "model": model_path,
            "metrics_json": os.path.join(args.out, "metrics.json"),
            "classification_report_txt": os.path.join(args.out, "classification_report.txt"),
            "confusion_matrix_png": cm_path,
            "test_predictions_csv": os.path.join(args.out, "test_predictions.csv"),
        },
        "metrics": metrics
    }, indent=2))


if __name__ == "__main__":
    main()
