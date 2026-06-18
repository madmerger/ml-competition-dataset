"""
Multi-faceted model validation using BCG FACET.

Analyses performed:
  1. SHAP-based feature importance (both models)
  2. Feature synergy matrix (BDT)
  3. Feature redundancy matrix (BDT)
  4. Redundancy / synergy dendrograms
  5. Univariate probability simulations for top features
  6. Bootstrap cross-validation stability analysis
  7. SHAP value distributions (beeswarm-style)

All figures are saved to outputs/facet_*.png and referenced in the final report.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from facet.data import Sample
from facet.data.partition import ContinuousRangePartitioner
from facet.inspection import NativeLearnerInspector
from facet.simulation import UnivariateProbabilitySimulator
from facet.simulation.viz import SimulationDrawer
from pytools.viz.dendrogram import DendrogramDrawer
from pytools.viz.matrix import MatrixDrawer
from sklearndf.classification import GradientBoostingClassifierDF
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_and_merge() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    ground_truth = pd.read_csv(DATA / "ground_truth.csv")
    machine_log = pd.read_csv(DATA / "machine_log.csv")

    sensor_cols = ["temperature_1", "temperature_2", "temperature_3", "pressure"]
    agg_dict: dict[str, list[str]] = {c: ["mean", "std", "min", "max"] for c in sensor_cols}
    agg_dict["maintenance_count"] = ["first"]

    ml_agg = machine_log.groupby(["line", "batch_count"]).agg(agg_dict)
    ml_agg.columns = ["_".join(col).strip("_") for col in ml_agg.columns]
    ml_agg = ml_agg.reset_index()

    train = train.merge(ml_agg, on=["line", "batch_count"], how="left")
    test = test.merge(ml_agg, on=["line", "batch_count"], how="left")

    train["target"] = (train["rank"] == "D").astype(int)
    test = test.merge(ground_truth[["product_id", "prediction"]], on="product_id", how="left")
    test = test.rename(columns={"prediction": "target"})

    feature_cols = [
        "line", "batch_count", "tray_no", "position", "maintenance_count_first",
    ] + [f"{s}_{a}" for s in sensor_cols for a in ["mean", "std", "min", "max"]]

    X_train = train[feature_cols].copy()
    y_train = train["target"].copy()
    X_test = test[feature_cols].copy()
    y_test = test["target"].copy()
    return X_train, y_train, X_test, y_test, feature_cols


def section_1_shap_inspection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
) -> None:
    """SHAP-based feature inspection using FACET's NativeLearnerInspector."""

    print("\n=== Section 1: SHAP-based Feature Inspection (BDT) ===")

    # sklearn GBC is used instead of XGBoost because FACET's TreeExplainer
    # requires SHAP interaction values in a format XGBoost does not provide.
    # Hyperparameters are aligned with the XGBoost model in train.py.
    gbc = GradientBoostingClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
        random_state=42,
    )
    gbc.fit(X_train, y_train)

    df_all = X_train.copy()
    df_all["target"] = y_train.values
    sample = Sample(observations=df_all, target_name="target", feature_names=feature_cols)

    inspector = NativeLearnerInspector(
        model=gbc, shap_interaction=True, n_jobs=-1,
    )
    inspector.fit(sample)

    # 1a. Feature importance
    importance = inspector.feature_importance()
    print("\nFeature importance (SHAP):")
    print(importance.to_string())

    fig, ax = plt.subplots(figsize=(9, 6))
    imp_sorted = importance.sort_values(ascending=True)
    ax.barh(range(len(imp_sorted)), imp_sorted.values)
    ax.set_yticks(range(len(imp_sorted)))
    ax.set_yticklabels(imp_sorted.index)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("FACET: SHAP Feature Importance (BDT)")
    fig.tight_layout()
    fig.savefig(OUT / "facet_shap_importance.png", dpi=150)
    plt.close(fig)

    # 1b. Synergy matrix
    print("\nComputing synergy matrix...")
    synergy = inspector.feature_synergy_matrix()
    fig, ax = plt.subplots(figsize=(12, 10))
    MatrixDrawer(style="matplot%").draw(synergy, title="Feature Synergy Matrix")
    plt.tight_layout()
    plt.savefig(OUT / "facet_synergy_matrix.png", dpi=150)
    plt.close("all")

    # 1c. Redundancy matrix
    print("Computing redundancy matrix...")
    redundancy = inspector.feature_redundancy_matrix()
    fig, ax = plt.subplots(figsize=(12, 10))
    MatrixDrawer(style="matplot%").draw(redundancy, title="Feature Redundancy Matrix")
    plt.tight_layout()
    plt.savefig(OUT / "facet_redundancy_matrix.png", dpi=150)
    plt.close("all")

    # 1d. Synergy dendrogram
    print("Computing synergy linkage...")
    synergy_linkage = inspector.feature_synergy_linkage()
    fig, ax = plt.subplots(figsize=(10, 6))
    DendrogramDrawer().draw(data=synergy_linkage, title="Feature Synergy Dendrogram")
    plt.tight_layout()
    plt.savefig(OUT / "facet_synergy_dendrogram.png", dpi=150)
    plt.close("all")

    # 1e. Redundancy dendrogram
    print("Computing redundancy linkage...")
    redundancy_linkage = inspector.feature_redundancy_linkage()
    fig, ax = plt.subplots(figsize=(10, 6))
    DendrogramDrawer().draw(data=redundancy_linkage, title="Feature Redundancy Dendrogram")
    plt.tight_layout()
    plt.savefig(OUT / "facet_redundancy_dendrogram.png", dpi=150)
    plt.close("all")

    # 1f. SHAP value distribution
    print("Computing SHAP values...")
    shap_values = inspector.shap_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    shap_abs_mean = shap_values.abs().mean().sort_values(ascending=True)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(shap_abs_mean)))
    ax.barh(range(len(shap_abs_mean)), shap_abs_mean.values, color=colors)
    ax.set_yticks(range(len(shap_abs_mean)))
    ax.set_yticklabels(shap_abs_mean.index)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Value Distribution (BDT)")
    fig.tight_layout()
    fig.savefig(OUT / "facet_shap_distribution.png", dpi=150)
    plt.close(fig)

    return gbc, sample


def section_2_simulation(
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
    feature_cols: list[str],
    top_features: list[str],
) -> None:
    """Univariate probability simulation for top features."""

    print("\n=== Section 2: Univariate Probability Simulation ===")
    print(f"Simulating top features: {top_features}")

    # Fit a DF-wrapped classifier for simulation
    gbc_df = GradientBoostingClassifierDF(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    gbc_df.fit(X_sub, y_sub)

    df_all = X_sub.copy()
    df_all["target"] = y_sub.values
    sample = Sample(observations=df_all, target_name="target", feature_names=feature_cols)

    partitioner = ContinuousRangePartitioner()

    for feat in top_features:
        print(f"  Simulating: {feat}")
        simulator = UnivariateProbabilitySimulator(
            model=gbc_df,
            sample=sample,
            n_jobs=-1,
        )
        simulation = simulator.simulate_feature(
            feature_name=feat, partitioner=partitioner,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        SimulationDrawer().draw(data=simulation, title=f"Probability Simulation: {feat}")
        plt.tight_layout()
        safe_name = feat.replace("/", "_")
        plt.savefig(OUT / f"facet_simulation_{safe_name}.png", dpi=150)
        plt.close("all")


def section_3_bootstrap_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Bootstrap cross-validation to assess model stability."""

    print("\n=== Section 3: Bootstrap Cross-Validation Stability ===")

    n_bootstrap = 50
    bdt_accs, bdt_aucs = [], []
    lr_accs, lr_aucs = [], []

    rng = np.random.RandomState(42)

    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_bs = X_train.iloc[idx]
        y_bs = y_train.iloc[idx]

        # BDT — use XGBClassifier to match train.py
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        xgb.fit(X_bs, y_bs)
        bdt_p = xgb.predict_proba(X_test)[:, 1]
        bdt_accs.append(accuracy_score(y_test, (bdt_p >= 0.5).astype(int)))
        bdt_aucs.append(roc_auc_score(y_test, bdt_p))

        # LR — fit a fresh scaler per bootstrap sample
        scaler = StandardScaler()
        X_bs_scaled = pd.DataFrame(
            scaler.fit_transform(X_bs), columns=feature_cols, index=X_bs.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=feature_cols, index=X_test.index,
        )
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_bs_scaled, y_bs)
        lr_p = lr.predict_proba(X_test_scaled)[:, 1]
        lr_accs.append(accuracy_score(y_test, (lr_p >= 0.5).astype(int)))
        lr_aucs.append(roc_auc_score(y_test, lr_p))

        if (i + 1) % 10 == 0:
            print(f"  Bootstrap iteration {i + 1}/{n_bootstrap}")

    results = pd.DataFrame({
        "BDT_Accuracy": bdt_accs,
        "BDT_AUC": bdt_aucs,
        "LR_Accuracy": lr_accs,
        "LR_AUC": lr_aucs,
    })

    # Plot stability
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(results["BDT_Accuracy"], bins=15, alpha=0.6, label="BDT", color="steelblue")
    axes[0].hist(results["LR_Accuracy"], bins=15, alpha=0.6, label="LR", color="coral")
    axes[0].axvline(results["BDT_Accuracy"].mean(), color="steelblue", linestyle="--")
    axes[0].axvline(results["LR_Accuracy"].mean(), color="coral", linestyle="--")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Bootstrap Accuracy Distribution")
    axes[0].legend()

    axes[1].hist(results["BDT_AUC"], bins=15, alpha=0.6, label="BDT", color="steelblue")
    axes[1].hist(results["LR_AUC"], bins=15, alpha=0.6, label="LR", color="coral")
    axes[1].axvline(results["BDT_AUC"].mean(), color="steelblue", linestyle="--")
    axes[1].axvline(results["LR_AUC"].mean(), color="coral", linestyle="--")
    axes[1].set_xlabel("AUC")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Bootstrap AUC Distribution")
    axes[1].legend()

    fig.suptitle("Model Stability: Bootstrap Cross-Validation (n=50)")
    fig.tight_layout()
    fig.savefig(OUT / "facet_bootstrap_stability.png", dpi=150)
    plt.close(fig)

    # Summary statistics
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Accuracy", "AUC", "AUC"],
        "Model": ["BDT", "LR", "BDT", "LR"],
        "Mean": [
            results["BDT_Accuracy"].mean(),
            results["LR_Accuracy"].mean(),
            results["BDT_AUC"].mean(),
            results["LR_AUC"].mean(),
        ],
        "Std": [
            results["BDT_Accuracy"].std(),
            results["LR_Accuracy"].std(),
            results["BDT_AUC"].std(),
            results["LR_AUC"].std(),
        ],
        "Min": [
            results["BDT_Accuracy"].min(),
            results["LR_Accuracy"].min(),
            results["BDT_AUC"].min(),
            results["LR_AUC"].min(),
        ],
        "Max": [
            results["BDT_Accuracy"].max(),
            results["LR_Accuracy"].max(),
            results["BDT_AUC"].max(),
            results["LR_AUC"].max(),
        ],
    })
    summary.to_csv(OUT / "facet_bootstrap_summary.csv", index=False)
    print("\nBootstrap stability summary:")
    print(summary.to_string(index=False))

    return results


def section_4_cross_val_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
) -> None:
    """Stratified K-Fold CV comparison of both models."""

    print("\n=== Section 4: Stratified K-Fold Cross-Validation ===")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )

    from sklearn.pipeline import Pipeline
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    bdt_scores = cross_val_score(xgb, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    lr_scores = cross_val_score(lr_pipe, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

    print(f"  BDT 5-fold AUC: {bdt_scores.mean():.4f} +/- {bdt_scores.std():.4f}")
    print(f"  LR  5-fold AUC: {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = [1, 2]
    bp = ax.boxplot(
        [bdt_scores, lr_scores], positions=positions, widths=0.4,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][1].set_facecolor("coral")
    ax.set_xticks(positions)
    ax.set_xticklabels(["BDT", "LR"])
    ax.set_ylabel("AUC")
    ax.set_title("5-Fold Stratified CV: AUC Comparison")
    for i, scores in enumerate([bdt_scores, lr_scores]):
        ax.scatter([positions[i]] * len(scores), scores, color="black", zorder=5, s=20)
    fig.tight_layout()
    fig.savefig(OUT / "facet_cv_comparison.png", dpi=150)
    plt.close(fig)


def section_5_feature_interaction(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
) -> None:
    """Feature interaction analysis."""

    print("\n=== Section 5: Feature Interaction Matrix ===")

    gbc = GradientBoostingClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    gbc.fit(X_train, y_train)

    df_all = X_train.copy()
    df_all["target"] = y_train.values
    sample = Sample(observations=df_all, target_name="target", feature_names=feature_cols)

    inspector = NativeLearnerInspector(model=gbc, shap_interaction=True, n_jobs=-1)
    inspector.fit(sample)

    interaction_matrix = inspector.feature_interaction_matrix()
    fig, ax = plt.subplots(figsize=(12, 10))
    MatrixDrawer(style="matplot%").draw(interaction_matrix, title="Feature Interaction Matrix")
    plt.tight_layout()
    plt.savefig(OUT / "facet_interaction_matrix.png", dpi=150)
    plt.close("all")

    association_matrix = inspector.feature_association_matrix()
    fig, ax = plt.subplots(figsize=(12, 10))
    MatrixDrawer(style="matplot%").draw(association_matrix, title="Feature Association Matrix")
    plt.tight_layout()
    plt.savefig(OUT / "facet_association_matrix.png", dpi=150)
    plt.close("all")


def main() -> None:
    print("Loading data...")
    X_train, y_train, X_test, y_test, feature_cols = load_and_merge()

    # Use a subsample for FACET analyses (computationally expensive)
    subsample_size = 5000
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_train), size=subsample_size, replace=False)
    X_sub = X_train.iloc[idx].reset_index(drop=True)
    y_sub = y_train.iloc[idx].reset_index(drop=True)

    print(f"Using subsample of {subsample_size} for FACET analysis")

    gbc, sample = section_1_shap_inspection(X_sub, y_sub, feature_cols)

    # Get top features for simulation
    inspector_tmp = NativeLearnerInspector(model=gbc, shap_interaction=False, n_jobs=-1)
    inspector_tmp.fit(sample)
    top_features = inspector_tmp.feature_importance().nlargest(5).index.tolist()

    section_2_simulation(X_sub, y_sub, feature_cols, top_features)
    section_3_bootstrap_stability(X_train, y_train, X_test, y_test, feature_cols)
    section_4_cross_val_comparison(X_train, y_train, feature_cols)
    section_5_feature_interaction(X_sub, y_sub, feature_cols)

    print("\n=== All FACET analyses complete. Outputs in outputs/ ===")


if __name__ == "__main__":
    main()
