"""
Train BDT (LightGBM) and Logistic Regression models to predict Rank-D
in the Panasonic ML competition dataset, then evaluate on test data.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load train, test, ground_truth and machine_log CSVs."""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    ground_truth = pd.read_csv(os.path.join(DATA_DIR, "ground_truth.csv"))
    machine_log = pd.read_csv(os.path.join(DATA_DIR, "machine_log.csv"))
    return train, test, ground_truth, machine_log


def create_machine_log_features(machine_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate machine_log by (line, batch_count) into statistical features."""
    sensor_cols = ["temperature_1", "temperature_2", "temperature_3", "pressure"]
    agg_funcs = ["mean", "std", "min", "max"]

    agg_dict = {col: agg_funcs for col in sensor_cols}
    agg_dict["maintenance_count"] = "max"
    agg_dict["process_time"] = "max"

    grouped = machine_log.groupby(["line", "batch_count"]).agg(agg_dict)
    grouped.columns = [
        f"{col}_{func}" if func != "" else col
        for col, func in grouped.columns
    ]
    grouped = grouped.reset_index()

    # Add range features
    for col in sensor_cols:
        grouped[f"{col}_range"] = grouped[f"{col}_max"] - grouped[f"{col}_min"]

    return grouped


def prepare_features(
    df: pd.DataFrame, ml_features: pd.DataFrame
) -> pd.DataFrame:
    """Merge product data with machine_log features."""
    merged = df.merge(ml_features, on=["line", "batch_count"], how="left")
    return merged


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature columns (excluding identifiers and target)."""
    exclude = ["product_id", "rank"]
    return [c for c in df.columns if c not in exclude]


def train_and_evaluate():
    """Main training and evaluation pipeline."""
    print("Loading data...")
    train, test, ground_truth, machine_log = load_data()

    print("Creating machine log features...")
    ml_features = create_machine_log_features(machine_log)

    print("Preparing training features...")
    train_merged = prepare_features(train, ml_features)
    test_merged = prepare_features(test, ml_features)

    # Binary target: rank D = 1, else = 0
    train_merged["target"] = (train_merged["rank"] == "D").astype(int)

    feature_cols = get_feature_columns(train_merged)
    feature_cols = [c for c in feature_cols if c != "target"]

    X_train = train_merged[feature_cols].values
    y_train = train_merged["target"].values

    # Merge test with ground_truth on product_id to ensure row alignment
    test_with_labels = test_merged.merge(
        ground_truth[["product_id", "prediction"]], on="product_id", how="inner"
    )
    X_test = test_with_labels[feature_cols].values
    y_test = test_with_labels["prediction"].values

    print(f"Features: {len(feature_cols)}")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train D ratio: {y_train.mean():.4f}")
    print(f"Test D ratio: {y_test.mean():.4f}")

    # --- Model 1: LightGBM (Boosted Decision Tree) ---
    print("\n=== Training LightGBM (Boosted Decision Tree) ===")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)

    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = (lgb_proba >= 0.5).astype(int)
    lgb_accuracy = accuracy_score(y_test, lgb_pred)
    lgb_auc = roc_auc_score(y_test, lgb_proba)

    print(f"LightGBM Accuracy: {lgb_accuracy:.4f}")
    print(f"LightGBM AUC: {lgb_auc:.4f}")

    # --- Model 2: Logistic Regression ---
    print("\n=== Training Logistic Regression ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle NaN (from missing machine_log entries)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

    lr_model = LogisticRegression(
        max_iter=1000, C=1.0, random_state=42, solver="lbfgs"
    )
    lr_model.fit(X_train_scaled, y_train)

    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_proba)

    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Logistic Regression AUC: {lr_auc:.4f}")

    # --- Results Summary ---
    results = {
        "LightGBM": {"Accuracy": lgb_accuracy, "AUC": lgb_auc},
        "LogisticRegression": {"Accuracy": lr_accuracy, "AUC": lr_auc},
    }
    print("\n=== Model Comparison ===")
    print(f"{'Model':<25} {'Accuracy':<12} {'AUC':<12}")
    print("-" * 49)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['Accuracy']:<12.4f} {metrics['AUC']:<12.4f}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save feature names for FACET analysis
    pd.DataFrame({"feature": feature_cols}).to_csv(
        os.path.join(OUTPUT_DIR, "feature_names.csv"), index=False
    )

    # Save train data for FACET
    train_for_facet = pd.DataFrame(X_train, columns=feature_cols)
    train_for_facet["target"] = y_train
    train_for_facet.to_csv(
        os.path.join(OUTPUT_DIR, "train_features.csv"), index=False
    )

    # --- Plot: Model Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results.keys())
    accuracies = [results[m]["Accuracy"] for m in models]
    aucs = [results[m]["AUC"] for m in models]

    axes[0].bar(models, accuracies, color=["#2196F3", "#FF9800"])
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.5, 1.0)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")

    axes[1].bar(models, aucs, color=["#2196F3", "#FF9800"])
    axes[1].set_title("AUC Comparison")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0.5, 1.0)
    for i, v in enumerate(aucs):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")

    plt.suptitle("Rank-D Prediction: Model Performance on Test Data")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # --- Plot: Feature Importance (LightGBM) ---
    importance = lgb_model.feature_importances_
    feat_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": importance}
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(feat_imp))
    sns.barplot(
        data=feat_imp.head(top_n), x="importance", y="feature", ax=ax
    )
    ax.set_title("LightGBM Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}/")
    return lgb_model, lr_model, scaler, feature_cols, results


if __name__ == "__main__":
    train_and_evaluate()
