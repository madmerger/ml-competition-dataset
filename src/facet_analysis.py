"""
Multi-faceted model validation using BCG FACET library.
Analyzes the LightGBM model for feature interactions, redundancy, and robustness.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from facet.data import Sample
from facet.inspection import LearnerInspector

from sklearndf.pipeline import ClassifierPipelineDF
from sklearndf.classification.extra import LGBMClassifierDF
from sklearndf.transformation import SimpleImputerDF

from pytools.viz.dendrogram import DendrogramDrawer
from pytools.viz.matrix import MatrixDrawer

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_prepared_data() -> tuple:
    """Load the pre-prepared training features."""
    train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train_features.csv"))
    feature_names = train_df.columns.drop("target").tolist()
    return train_df, feature_names


def run_facet_analysis():
    """Run comprehensive FACET analysis on the LightGBM model."""
    print("Loading prepared data...")
    train_df, feature_names = load_prepared_data()

    # Use a subsample for faster FACET analysis (SHAP is expensive)
    sample_size = min(5000, len(train_df))
    train_sample = train_df.sample(n=sample_size, random_state=42).reset_index(
        drop=True
    )

    # Create FACET Sample object
    sample = Sample(
        observations=train_sample,
        feature_names=feature_names,
        target_name="target",
    )

    print(f"Sample size: {sample_size}, Features: {len(feature_names)}")

    # --- Step 1: Build Pipeline and Fit ---
    print("\n=== Step 1: Build and Fit LightGBM Pipeline ===")

    preprocessing = SimpleImputerDF(strategy="median")

    lgbm_pipeline = ClassifierPipelineDF(
        preprocessing=preprocessing,
        classifier=LGBMClassifierDF(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
    )

    # Fit the pipeline on the sample
    X_sample = train_sample[feature_names]
    y_sample = train_sample["target"]
    lgbm_pipeline.fit(X_sample, y_sample)
    print("Pipeline fitted successfully.")

    # --- Step 2: LearnerInspector for SHAP-based analysis ---
    print("\n=== Step 2: SHAP-based Model Inspection ===")

    inspector = LearnerInspector(
        model=lgbm_pipeline,
        n_jobs=-1,
        verbose=0,
    ).fit(sample)

    # --- 2a: Feature Importance ---
    print("Computing feature importance...")
    f_importance = inspector.feature_importance()

    fig, ax = plt.subplots(figsize=(10, 8))
    f_importance.sort_values().plot.barh(ax=ax)
    ax.set_title("FACET Feature Importance (SHAP-based)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "facet_feature_importance.png"), dpi=150
    )
    plt.close()
    print("  -> facet_feature_importance.png saved")

    # --- 2b: SHAP Summary Plot ---
    print("Generating SHAP summary plot...")
    shap_data = inspector.shap_plot_data()

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values=shap_data.shap_values,
        features=shap_data.features,
        show=False,
        plot_size=(12, 8),
    )
    plt.title("SHAP Summary Plot - Rank D Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "facet_shap_summary.png"), dpi=150)
    plt.close()
    print("  -> facet_shap_summary.png saved")

    # --- 2c: Feature Synergy Matrix ---
    print("Computing feature synergy matrix...")
    synergy_matrix = inspector.feature_synergy_matrix()

    fig, ax = plt.subplots(figsize=(12, 10))
    MatrixDrawer(style="matplot%").draw(
        synergy_matrix, title="Feature Synergies"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "facet_synergy_matrix.png"), dpi=150)
    plt.close()
    print("  -> facet_synergy_matrix.png saved")

    # --- 2d: Feature Redundancy Matrix ---
    print("Computing feature redundancy matrix...")
    redundancy_matrix = inspector.feature_redundancy_matrix()

    fig, ax = plt.subplots(figsize=(12, 10))
    MatrixDrawer(style="matplot%").draw(
        redundancy_matrix, title="Feature Redundancies"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "facet_redundancy_matrix.png"), dpi=150
    )
    plt.close()
    print("  -> facet_redundancy_matrix.png saved")

    # --- 2e: Redundancy Dendrogram ---
    print("Computing redundancy linkage dendrogram...")
    dd_redundancy = inspector.feature_redundancy_linkage()

    fig, ax = plt.subplots(figsize=(14, 6))
    DendrogramDrawer().draw(
        title="Feature Redundancy Linkage", data=dd_redundancy
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "facet_redundancy_dendrogram.png"), dpi=150
    )
    plt.close()
    print("  -> facet_redundancy_dendrogram.png saved")

    # --- 2f: Feature Association Matrix ---
    print("Computing feature association matrix...")
    association_matrix = inspector.feature_association_matrix()

    fig, ax = plt.subplots(figsize=(12, 10))
    MatrixDrawer(style="matplot%").draw(
        association_matrix, title="Feature Associations (Synergy + Redundancy)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "facet_association_matrix.png"), dpi=150
    )
    plt.close()
    print("  -> facet_association_matrix.png saved")

    # --- Step 3: Robustness via Bootstrap Performance Distribution ---
    print("\n=== Step 3: Bootstrap Robustness Analysis ===")

    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score

    n_bootstrap = 30
    auc_scores = []
    acc_scores = []

    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        # Bootstrap sample from training data
        idx = rng.choice(len(train_sample), size=len(train_sample), replace=True)
        oob_idx = np.setdiff1d(np.arange(len(train_sample)), idx)

        if len(oob_idx) < 50:
            continue

        X_boot = train_sample.iloc[idx][feature_names]
        y_boot = train_sample.iloc[idx]["target"]
        X_oob = train_sample.iloc[oob_idx][feature_names]
        y_oob = train_sample.iloc[oob_idx]["target"]

        # Impute NaN with pre-imputation median
        boot_median = X_boot.median()
        X_boot = X_boot.fillna(boot_median)
        X_oob = X_oob.fillna(boot_median)

        model = LGBMClassifier(
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
        model.fit(X_boot, y_boot)
        proba = model.predict_proba(X_oob)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc_scores.append(roc_auc_score(y_oob, proba))
        acc_scores.append(accuracy_score(y_oob, pred))

    auc_scores = np.array(auc_scores)
    acc_scores = np.array(acc_scores)

    print(f"Bootstrap AUC: {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")
    print(
        f"Bootstrap Accuracy: {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}"
    )

    # Plot bootstrap distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(auc_scores, bins=15, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].axvline(
        auc_scores.mean(), color="red", linestyle="--", label=f"Mean={auc_scores.mean():.4f}"
    )
    axes[0].set_title("Bootstrap AUC Distribution (LightGBM)")
    axes[0].set_xlabel("AUC")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(acc_scores, bins=15, color="#FF9800", edgecolor="white", alpha=0.8)
    axes[1].axvline(
        acc_scores.mean(), color="red", linestyle="--", label=f"Mean={acc_scores.mean():.4f}"
    )
    axes[1].set_title("Bootstrap Accuracy Distribution (LightGBM)")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.suptitle("Model Robustness: Bootstrap Performance Distribution")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "facet_bootstrap_robustness.png"), dpi=150
    )
    plt.close()
    print("  -> facet_bootstrap_robustness.png saved")

    # --- Step 4: Feature Stability Analysis ---
    print("\n=== Step 4: Feature Stability Analysis ===")

    # Check if top features are consistent across bootstrap runs
    importance_per_boot = []
    for i in range(min(10, n_bootstrap)):
        idx = rng.choice(len(train_sample), size=len(train_sample), replace=True)
        X_boot = train_sample.iloc[idx][feature_names]
        X_boot = X_boot.fillna(X_boot.median())
        y_boot = train_sample.iloc[idx]["target"]

        model = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=i,
            verbose=-1,
        )
        model.fit(X_boot, y_boot)
        importance_per_boot.append(model.feature_importances_)

    importance_array = np.array(importance_per_boot)
    importance_mean = importance_array.mean(axis=0)
    importance_std = importance_array.std(axis=0)

    stability_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance_mean,
            "importance_std": importance_std,
            "cv": importance_std / (importance_mean + 1e-10),
        }
    ).sort_values("importance_mean", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_stability = stability_df.head(15)
    ax.barh(
        range(len(top_stability)),
        top_stability["importance_mean"],
        xerr=top_stability["importance_std"],
        color="#4CAF50",
        alpha=0.8,
        capsize=3,
    )
    ax.set_yticks(range(len(top_stability)))
    ax.set_yticklabels(top_stability["feature"])
    ax.set_xlabel("Feature Importance (mean +/- std across bootstraps)")
    ax.set_title("Feature Importance Stability (Top 15)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "facet_feature_stability.png"), dpi=150
    )
    plt.close()
    print("  -> facet_feature_stability.png saved")

    # --- Save analysis summary ---
    summary = {
        "bootstrap_auc_mean": float(auc_scores.mean()),
        "bootstrap_auc_std": float(auc_scores.std()),
        "bootstrap_accuracy_mean": float(acc_scores.mean()),
        "bootstrap_accuracy_std": float(acc_scores.std()),
        "n_features": len(feature_names),
        "top_5_features": stability_df.head(5)["feature"].tolist(),
        "top_5_importance_cv": stability_df.head(5)["cv"].tolist(),
    }

    import json

    with open(os.path.join(OUTPUT_DIR, "facet_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FACET Analysis Complete ===")
    print(f"All outputs saved to {OUTPUT_DIR}/")

    return summary, f_importance, synergy_matrix, redundancy_matrix


if __name__ == "__main__":
    run_facet_analysis()
