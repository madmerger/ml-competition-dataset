"""
加速試験ランクD予測モデル
==========================
製造工程データ（train.csv, machine_log.csv）を用いて、加速試験結果がランクDとなるかどうかを
連続値（確率）として予測する。

モデル:
  1. Boosted Decision Tree (LightGBM)
  2. Linear Regression (Logistic Regression)

評価指標: Accuracy, AUC (on test data using ground_truth.csv)

参考: https://tech-ai.panasonic.com/jp/blog_page.html?id=20220805
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_data():
    """CSVファイルを読み込む"""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    machine_log = pd.read_csv(os.path.join(DATA_DIR, "machine_log.csv"))
    ground_truth = pd.read_csv(os.path.join(DATA_DIR, "ground_truth.csv"))
    return train, test, machine_log, ground_truth


def create_target(train):
    """ランクDかどうかの二値ターゲットを作成（D=1, その他=0）"""
    train["target"] = (train["rank"] == "D").astype(int)
    return train


def aggregate_machine_log(machine_log):
    """machine_logを(line, batch_count)でグルーピングし、統計量を集約する"""
    numeric_cols = ["process_time", "temperature_1", "temperature_2", "temperature_3", "pressure"]

    agg_funcs = {}
    for col in numeric_cols:
        agg_funcs[col] = ["mean", "std", "min", "max", "median"]

    agg_funcs["maintenance_count"] = ["max"]
    agg_funcs["datetime"] = ["count"]

    log_agg = machine_log.groupby(["line", "batch_count"]).agg(agg_funcs)
    log_agg.columns = ["_".join(col).strip() for col in log_agg.columns.values]
    log_agg = log_agg.rename(columns={"datetime_count": "log_count"})
    log_agg = log_agg.reset_index()

    return log_agg


def build_features(df, log_agg):
    """特徴量を構築する"""
    merged = df.merge(log_agg, on=["line", "batch_count"], how="left")

    feature_cols = [col for col in merged.columns if col not in [
        "product_id", "rank", "target", "line", "batch_count"
    ]]

    merged[feature_cols] = merged[feature_cols].fillna(0)

    return merged, feature_cols


def train_lightgbm(X_train, y_train):
    """LightGBM (Boosted Decision Tree) モデルを学習する"""
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=500)
    return model


def train_logistic_regression(X_train, y_train):
    """Logistic Regression (線形回帰ベース) モデルを学習する"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_scaled, y_train)
    return model, scaler


def evaluate_model(y_true, y_pred_proba, model_name, threshold=0.5):
    """モデルの評価（Accuracy, AUC）"""
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred_proba)
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"{'='*50}")
    return acc, auc


def plot_roc_curves(y_true, preds_dict, output_path):
    """ROC曲線を描画して保存する"""
    plt.figure(figsize=(8, 6))
    for name, y_pred in preds_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_val = roc_auc_score(y_true, y_pred)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nROC curve saved to: {output_path}")


def plot_comparison_bar(results, output_path):
    """AccuracyとAUCの比較棒グラフを描画する"""
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]
    aucs = [results[m]["auc"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, aucs, width, label="AUC", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy vs AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("加速試験ランクD予測モデル")
    print("=" * 60)

    # データ読み込み
    print("\n[1/6] データ読み込み中...")
    train, test, machine_log, ground_truth = load_data()
    print(f"  train: {train.shape}, test: {test.shape}")
    print(f"  machine_log: {machine_log.shape}")
    print(f"  ground_truth: {ground_truth.shape}")

    # ターゲット作成
    print("\n[2/6] ターゲット変数作成中...")
    train = create_target(train)
    print(f"  rank D率 (train): {train['target'].mean():.4f} ({train['target'].sum()}/{len(train)})")

    # 特徴量生成
    print("\n[3/6] machine_logから特徴量を集約中...")
    log_agg = aggregate_machine_log(machine_log)
    print(f"  集約後のログ: {log_agg.shape}")

    train_merged, feature_cols = build_features(train, log_agg)
    test_merged, _ = build_features(test, log_agg)
    print(f"  特徴量数: {len(feature_cols)}")
    print(f"  特徴量: {feature_cols}")

    X_train = train_merged[feature_cols].values
    y_train = train_merged["target"].values
    X_test = test_merged[feature_cols].values

    # テストデータの正解ラベルを取得
    test_gt = test_merged[["product_id"]].merge(ground_truth, on="product_id", how="left")
    y_test = test_gt["prediction"].values

    print(f"\n  train samples: {len(X_train)}, test samples: {len(X_test)}")
    print(f"  test rank D率: {y_test.mean():.4f} ({int(y_test.sum())}/{len(y_test)})")

    # モデル1: LightGBM (Boosted Decision Tree)
    print("\n[4/6] LightGBM (Boosted Decision Tree) 学習中...")
    lgbm_model = train_lightgbm(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_test)

    # モデル2: Logistic Regression (線形回帰モデル)
    print("\n[5/6] Logistic Regression (線形回帰モデル) 学習中...")
    lr_model, scaler = train_logistic_regression(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]

    # 評価
    print("\n[6/6] テストデータで評価中...")
    lgbm_acc, lgbm_auc = evaluate_model(y_test, lgbm_pred, "Boosted Decision Tree (LightGBM)")
    lr_acc, lr_auc = evaluate_model(y_test, lr_pred, "Linear Regression (Logistic Regression)")

    # 結果比較テーブル
    results = {
        "Boosted Decision Tree\n(LightGBM)": {"accuracy": lgbm_acc, "auc": lgbm_auc},
        "Linear Regression\n(Logistic Regression)": {"accuracy": lr_acc, "auc": lr_auc},
    }

    print("\n" + "=" * 60)
    print("モデル比較結果")
    print("=" * 60)
    print(f"{'モデル':<45} {'Accuracy':>10} {'AUC':>10}")
    print("-" * 65)
    print(f"{'Boosted Decision Tree (LightGBM)':<45} {lgbm_acc:>10.4f} {lgbm_auc:>10.4f}")
    print(f"{'Linear Regression (Logistic Regression)':<45} {lr_acc:>10.4f} {lr_auc:>10.4f}")
    print("=" * 60)

    # グラフ出力
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    bar_path = os.path.join(OUTPUT_DIR, "model_comparison.png")

    plot_roc_curves(
        y_test,
        {
            "Boosted Decision Tree (LightGBM)": lgbm_pred,
            "Linear Regression (Logistic Regression)": lr_pred,
        },
        roc_path,
    )
    plot_comparison_bar(results, bar_path)

    # 予測結果をCSV出力
    submission_lgbm = pd.DataFrame({
        "product_id": test_merged["product_id"],
        "prediction": lgbm_pred,
    })
    submission_lr = pd.DataFrame({
        "product_id": test_merged["product_id"],
        "prediction": lr_pred,
    })
    submission_lgbm.to_csv(os.path.join(OUTPUT_DIR, "submission_lightgbm.csv"), index=False)
    submission_lr.to_csv(os.path.join(OUTPUT_DIR, "submission_logistic_regression.csv"), index=False)

    # 特徴量重要度（LightGBM）
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": lgbm_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance["feature"][:20], importance["importance"][:20])
    plt.xlabel("Feature Importance (Gain)")
    plt.title("LightGBM Feature Importance (Top 20)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    importance_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(importance_path, dpi=150)
    plt.close()
    print(f"Feature importance chart saved to: {importance_path}")

    print("\n完了！出力ファイルは output/ ディレクトリに保存されました。")


if __name__ == "__main__":
    main()
