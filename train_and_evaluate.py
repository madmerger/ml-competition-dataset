"""
加速試験ランクD予測モデル: Boosted Decision Tree vs 線形回帰

train.csv / test.csv と machine_log.csv を結合し、
加速試験の結果がランクDかどうかを連続値で予測するモデルを構築する。
Boosted Decision Tree (LightGBM) と線形回帰 (Logistic Regression) で
testデータに対する Accuracy を比較する。

参考: https://tech-ai.panasonic.com/jp/blog_page.html?id=20220805
"""

import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# 1. データ読み込み
# ---------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    machine_log = pd.read_csv(os.path.join(DATA_DIR, "machine_log.csv"))
    ground_truth = pd.read_csv(os.path.join(DATA_DIR, "ground_truth.csv"))
    return train, test, machine_log, ground_truth


# ---------------------------------------------------------------------------
# 2. machine_log から特徴量を集約 (line, batch_count ごと)
# ---------------------------------------------------------------------------
def aggregate_machine_log(machine_log: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = ["temperature_1", "temperature_2", "temperature_3", "pressure"]
    agg_funcs = ["mean", "std", "min", "max"]

    agg_dict = {col: agg_funcs for col in sensor_cols}
    agg_dict["process_time"] = ["max"]
    agg_dict["maintenance_count"] = ["max"]

    grouped = machine_log.groupby(["line", "batch_count"]).agg(agg_dict)
    grouped.columns = ["_".join(col).strip() for col in grouped.columns]
    grouped = grouped.reset_index()

    # temperature_1/2/3 の差分特徴量
    for func in agg_funcs:
        grouped[f"temp_diff_1_2_{func}"] = (
            grouped[f"temperature_1_{func}"] - grouped[f"temperature_2_{func}"]
        )
        grouped[f"temp_diff_2_3_{func}"] = (
            grouped[f"temperature_2_{func}"] - grouped[f"temperature_3_{func}"]
        )

    return grouped


# ---------------------------------------------------------------------------
# 3. train/test にマージして特徴量行列を構築
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, log_features: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(log_features, on=["line", "batch_count"], how="left")
    return merged


# ---------------------------------------------------------------------------
# 4. メイン処理
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("加速試験ランクD予測: Boosted Decision Tree vs 線形回帰")
    print("=" * 60)

    # --- データ読み込み ---
    train, test, machine_log, ground_truth = load_data()
    print(f"\ntrain: {len(train)} rows, test: {len(test)} rows")
    print(f"machine_log: {len(machine_log)} rows")

    # --- ターゲット作成 (rank == 'D' → 1, それ以外 → 0) ---
    train["target"] = (train["rank"] == "D").astype(int)
    print("\nターゲット分布 (train):")
    print(train["target"].value_counts().to_string())

    # --- machine_log 集約 ---
    print("\nmachine_log の特徴量集約中...")
    log_features = aggregate_machine_log(machine_log)
    print(f"集約後: {len(log_features)} batch 行")

    # --- 特徴量マージ ---
    train_merged = build_features(train, log_features)
    test_merged = build_features(test, log_features)

    # --- 特徴量カラムの定義 ---
    feature_cols = [
        col
        for col in train_merged.columns
        if col not in ["product_id", "rank", "target"]
    ]
    print(f"\n使用する特徴量数: {len(feature_cols)}")
    print(f"特徴量一覧: {feature_cols}")

    X_train = train_merged[feature_cols].values
    y_train = train_merged["target"].values
    X_test = test_merged[feature_cols].values

    # --- ground_truth からテストデータの正解を取得 ---
    test_with_gt = test_merged.merge(
        ground_truth[["product_id", "prediction"]],
        on="product_id",
        how="left",
    )
    y_test = test_with_gt["prediction"].values

    print("\nテストデータの正解分布:")
    print(f"  ランクD (1): {int(y_test.sum())}")
    print(f"  ランクD以外 (0): {int(len(y_test) - y_test.sum())}")

    # ===================================================================
    # モデル1: Boosted Decision Tree (LightGBM)
    # ===================================================================
    print("\n" + "=" * 60)
    print("モデル1: Boosted Decision Tree (LightGBM)")
    print("=" * 60)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "random_state": 42,
    }

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)

    # 連続値予測 (確率)
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    # 閾値0.5で二値化
    lgb_pred = (lgb_proba >= 0.5).astype(int)

    lgb_accuracy = accuracy_score(y_test, lgb_pred)
    lgb_auc = roc_auc_score(y_test, lgb_proba)

    print("\n  連続値予測の統計:")
    print(f"    平均: {lgb_proba.mean():.4f}")
    print(f"    標準偏差: {lgb_proba.std():.4f}")
    print(f"    最小: {lgb_proba.min():.4f}")
    print(f"    最大: {lgb_proba.max():.4f}")
    print(f"\n  Accuracy: {lgb_accuracy:.4f}")
    print(f"  AUC: {lgb_auc:.4f}")

    # 特徴量重要度
    importance = lgb_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print("\n  特徴量重要度 (上位10):")
    for feat, imp in feat_imp[:10]:
        print(f"    {feat}: {imp}")

    # ===================================================================
    # モデル2: 線形回帰 (Logistic Regression)
    # ===================================================================
    print("\n" + "=" * 60)
    print("モデル2: 線形回帰 (Logistic Regression)")
    print("=" * 60)

    # 欠損値を中央値で補完
    X_train_lr = np.nan_to_num(X_train, nan=0.0)
    X_test_lr = np.nan_to_num(X_test, nan=0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_lr)
    X_test_scaled = scaler.transform(X_test_lr)

    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver="lbfgs")
    lr_model.fit(X_train_scaled, y_train)

    # 連続値予測 (確率)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    # 閾値0.5で二値化
    lr_pred = (lr_proba >= 0.5).astype(int)

    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_proba)

    print("\n  連続値予測の統計:")
    print(f"    平均: {lr_proba.mean():.4f}")
    print(f"    標準偏差: {lr_proba.std():.4f}")
    print(f"    最小: {lr_proba.min():.4f}")
    print(f"    最大: {lr_proba.max():.4f}")
    print(f"\n  Accuracy: {lr_accuracy:.4f}")
    print(f"  AUC: {lr_auc:.4f}")

    # ===================================================================
    # 結果比較
    # ===================================================================
    print("\n" + "=" * 60)
    print("結果比較")
    print("=" * 60)

    results = pd.DataFrame(
        {
            "モデル": [
                "Boosted Decision Tree (LightGBM)",
                "線形回帰 (Logistic Regression)",
            ],
            "Accuracy": [lgb_accuracy, lr_accuracy],
            "AUC": [lgb_auc, lr_auc],
        }
    )
    print("\n" + results.to_string(index=False))

    diff = lgb_accuracy - lr_accuracy
    if diff > 0:
        print(f"\n→ Boosted Decision Tree が線形回帰より Accuracy が {diff:.4f} 高い")
    elif diff < 0:
        print(f"\n→ 線形回帰が Boosted Decision Tree より Accuracy が {-diff:.4f} 高い")
    else:
        print("\n→ 両モデルの Accuracy は同じ")

    # --- 予測結果をCSVに出力 ---
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    lgb_submission = pd.DataFrame(
        {"product_id": test["product_id"], "prediction": lgb_proba}
    )
    lgb_submission.to_csv(os.path.join(output_dir, "submission_lgb.csv"), index=False)

    lr_submission = pd.DataFrame(
        {"product_id": test["product_id"], "prediction": lr_proba}
    )
    lr_submission.to_csv(os.path.join(output_dir, "submission_lr.csv"), index=False)

    print(f"\n予測結果を {output_dir}/ に保存しました")
    print("  - submission_lgb.csv (Boosted Decision Tree)")
    print("  - submission_lr.csv (線形回帰)")


if __name__ == "__main__":
    main()
