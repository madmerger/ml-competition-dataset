"""
Train BDT (XGBoost) and Logistic Regression models to predict acceleration-test
rank-D on the Panasonic ML competition dataset.

Outputs (written to outputs/):
  - model_comparison.csv          Accuracy & AUC per model
  - confusion_matrix_bdt.png      Confusion matrix for BDT
  - confusion_matrix_lr.png       Confusion matrix for LR
  - roc_curve_comparison.png      ROC curves overlay
  - feature_importance_bdt.png    XGBoost feature importance
  - bdt_model.json                Serialised XGBoost model
  - lr_model.pkl                  Serialised LR model
  - predictions_bdt.csv           Continuous predictions (BDT)
  - predictions_lr.csv            Continuous predictions (LR)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_and_merge() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load CSVs and create features by joining machine_log aggregates."""

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    ground_truth = pd.read_csv(DATA / "ground_truth.csv")
    machine_log = pd.read_csv(DATA / "machine_log.csv")

    # Aggregate machine_log per (line, batch_count)
    sensor_cols = ["temperature_1", "temperature_2", "temperature_3", "pressure"]
    agg_dict: dict[str, list[str]] = {c: ["mean", "std", "min", "max"] for c in sensor_cols}
    agg_dict["maintenance_count"] = ["first"]

    ml_agg = machine_log.groupby(["line", "batch_count"]).agg(agg_dict)
    ml_agg.columns = ["_".join(col).strip("_") for col in ml_agg.columns]
    ml_agg = ml_agg.reset_index()

    # Merge
    train = train.merge(ml_agg, on=["line", "batch_count"], how="left")
    test = test.merge(ml_agg, on=["line", "batch_count"], how="left")

    # Binary target: rank D = 1
    train["target"] = (train["rank"] == "D").astype(int)

    # Align test ground truth
    test = test.merge(ground_truth[["product_id", "prediction"]], on="product_id", how="left")
    test = test.rename(columns={"prediction": "target"})

    feature_cols = [
        "line",
        "batch_count",
        "tray_no",
        "position",
        "maintenance_count_first",
    ] + [
        f"{s}_{a}" for s in sensor_cols for a in ["mean", "std", "min", "max"]
    ]

    X_train = train[feature_cols].copy()
    y_train = train["target"].copy()
    X_test = test[feature_cols].copy()
    y_test = test["target"].copy()

    return X_train, y_train, X_test, y_test, feature_cols


def train_bdt(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[XGBClassifier, np.ndarray]:
    """Train XGBoost (Boosted Decision Tree) and return model + test probabilities."""

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return model, proba


def train_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    """Train Logistic Regression and return model + test probabilities."""

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr_scaled, y_train)
    proba = model.predict_proba(X_te_scaled)[:, 1]
    return model, scaler, proba


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not D", "D"])
    ax.set_yticklabels(["Not D", "D"])
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    probas: dict[str, np.ndarray],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_val = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model: XGBClassifier, feature_names: list[str], path: Path) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading and merging data...")
    X_train, y_train, X_test, y_test, feature_cols = load_and_merge()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Train target rate: {y_train.mean():.4f}")
    print(f"  Test  target rate: {y_test.mean():.4f}")

    # --- BDT ---
    print("\nTraining Boosted Decision Tree (XGBoost)...")
    bdt_model, bdt_proba = train_bdt(X_train, y_train, X_test)
    bdt_pred = (bdt_proba >= 0.5).astype(int)
    bdt_acc = accuracy_score(y_test, bdt_pred)
    bdt_auc = roc_auc_score(y_test, bdt_proba)
    print(f"  BDT  Accuracy: {bdt_acc:.4f}  AUC: {bdt_auc:.4f}")

    # --- LR ---
    print("\nTraining Logistic Regression...")
    lr_model, lr_scaler, lr_proba = train_lr(X_train, y_train, X_test)
    lr_pred = (lr_proba >= 0.5).astype(int)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_proba)
    print(f"  LR   Accuracy: {lr_acc:.4f}  AUC: {lr_auc:.4f}")

    # --- Comparison table ---
    comparison = pd.DataFrame({
        "Model": ["Boosted Decision Tree (XGBoost)", "Logistic Regression"],
        "Accuracy": [bdt_acc, lr_acc],
        "AUC": [bdt_auc, lr_auc],
    })
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))

    # --- Plots ---
    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, bdt_pred, "BDT Confusion Matrix", OUT / "confusion_matrix_bdt.png")
    plot_confusion_matrix(y_test, lr_pred, "LR Confusion Matrix", OUT / "confusion_matrix_lr.png")
    plot_roc_curves(y_test, {"BDT": bdt_proba, "LR": lr_proba}, OUT / "roc_curve_comparison.png")
    plot_feature_importance(bdt_model, feature_cols, OUT / "feature_importance_bdt.png")

    # --- Save models ---
    bdt_model.save_model(str(OUT / "bdt_model.json"))
    with open(OUT / "lr_model.pkl", "wb") as f:
        pickle.dump({"model": lr_model, "scaler": lr_scaler}, f)

    # --- Save predictions ---
    pd.DataFrame({"proba_D": bdt_proba}).to_csv(OUT / "predictions_bdt.csv", index=False)
    pd.DataFrame({"proba_D": lr_proba}).to_csv(OUT / "predictions_lr.csv", index=False)

    print("\nAll outputs written to outputs/")


if __name__ == "__main__":
    main()
