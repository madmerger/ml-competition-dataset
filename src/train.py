"""Train and compare two models that predict whether a product's accelerated
test result is rank D.

The target is a binary indicator (``rank == "D"``) but every model emits a
*continuous* score in ``[0, 1]`` (a probability-like value), as requested.
Two model families are compared:

* Boosted Decision Tree  -> LightGBM gradient boosting regressor
* Linear regression      -> ordinary least squares (scikit-learn)

Test accuracy is computed against ``ground_truth.csv`` by thresholding the
continuous prediction at 0.5.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_batch_features, build_features, feature_columns

THRESHOLD = 0.5


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the CSVs, extracting them from ``data.zip`` on first use."""
    needed = ["train.csv", "test.csv", "machine_log.csv", "ground_truth.csv"]
    if not all((data_dir / name).exists() for name in needed):
        zip_path = data_dir.parent / "data.zip"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)

    return {name.replace(".csv", ""): pd.read_csv(data_dir / name) for name in needed}


def prepare(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build train/test feature frames and the shared feature-column list."""
    batch_features = build_batch_features(data["machine_log"])

    train = build_features(data["train"], batch_features)
    train["is_rank_d"] = (train["rank"] == "D").astype(int)

    test = build_features(data["test"], batch_features)

    features = feature_columns(train)
    return train, test, features


def train_boosted_tree(x: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x, y)
    return model


def train_linear(x: pd.DataFrame, y: pd.Series):
    # Impute first: a left-join with the machine log (or a single-row batch's
    # std) can leave NaNs, which the linear pipeline cannot handle natively.
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()
    )
    model.fit(x, y)
    return model


def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Clip scores to [0, 1], threshold at 0.5 and report accuracy / AUC."""
    y_score = np.clip(y_score, 0.0, 1.0)
    y_pred = (y_score >= THRESHOLD).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
    }
    print(
        f"[{name:<20}] accuracy={metrics['accuracy']:.4f}  auc={metrics['auc']:.4f}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data_dir)
    train, test, features = prepare(data)

    x_train, y_train = train[features], train["is_rank_d"]
    x_test = test[features]

    ground_truth = data["ground_truth"].set_index("product_id")
    y_true = ground_truth.loc[test["product_id"], "prediction"].to_numpy()
    split = ground_truth.loc[test["product_id"], "split"].to_numpy()

    models = {
        "BoostedDecisionTree": train_boosted_tree,
        "LinearRegression": train_linear,
    }

    summary_rows = []
    for name, trainer in models.items():
        model = trainer(x_train, y_train)
        scores = np.clip(np.asarray(model.predict(x_test), dtype=float), 0.0, 1.0)

        print(f"\n=== {name} ===")
        overall = evaluate(f"{name} (all)", y_true, scores)
        public = evaluate(f"{name} (public)", y_true[split == 0], scores[split == 0])
        private = evaluate(f"{name} (private)", y_true[split == 1], scores[split == 1])

        pd.DataFrame({"product_id": test["product_id"], "prediction": scores}).to_csv(
            args.output_dir / f"submission_{name}.csv", index=False
        )
        summary_rows.append(
            {
                "model": name,
                "accuracy_all": overall["accuracy"],
                "accuracy_public": public["accuracy"],
                "accuracy_private": private["accuracy"],
                "auc_all": overall["auc"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "model_comparison.csv", index=False)
    print("\n=== Test accuracy comparison ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
