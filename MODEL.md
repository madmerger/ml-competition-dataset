# ランクD予測モデル

加速試験の結果が **ランクD** となるかどうかを予測し、`[0, 1]` の連続値（確率的スコア）として
出力する機械学習モデルです。Boosted Decision Tree と線形回帰モデルの2種類を学習し、
`test` データに対する Accuracy を比較します。

参考: [Panasonic 技術ブログ（社内MLコンペ）](https://tech-ai.panasonic.com/jp/blog_page.html?id=20220805)

## タスク設定

- **目的変数**: `rank == "D"` を 1、それ以外（A/B/C）を 0 とした二値ラベル。
- **出力**: 各モデルとも `[0, 1]` の連続値スコアを出力（要件どおり）。
- **評価**: `ground_truth.csv` の正解と比較し、スコアを 0.5 で閾値処理して Accuracy を算出。
  公開（public）/非公開（private）の split 別、および全体で評価します。

## 特徴量エンジニアリング

各製品は製造バッチ `(line, batch_count)` に属し、`machine_log.csv` には1バッチあたり
20ステップ分のセンサ時系列（温度3系統・圧力）が記録されています。
このバッチ単位の統計量を製品行に結合します（`src/features.py`）。

- 製品レベル: `line`, `tray_no`, `position`
- バッチレベル（センサごと）: 平均 / 標準偏差 / 最小 / 最大 / レンジ / トレンド（最終値 − 初期値）
- バッチレベル: `maintenance_count`（メンテからの経過＝摩耗の指標）

`batch_count` は `(line, batch_count)` でのセンサ結合キーとしてのみ使用し、
モデル入力からは除外しています（train は小さい値、test は約10000以上で単調増加する
インデックスのため、入力にすると分布外への外挿になり特に線形モデルが不安定になるため）。
線形モデル側は欠損混入時にも壊れないよう `SimpleImputer(median)` を前段に入れています。

## モデル

| 種別 | 実装 |
| --- | --- |
| Boosted Decision Tree | LightGBM（`LGBMRegressor`, `objective="binary"`） |
| 線形回帰 | scikit-learn `LinearRegression`（`StandardScaler` で標準化） |

## 実行方法

```bash
pip install -r requirements.txt
python src/train.py
```

`data/` が無い場合は `data.zip` から自動展開します。結果は `outputs/` に出力されます
（`submission_*.csv`, `model_comparison.csv`）。

## 結果（test データ Accuracy）

| モデル | Accuracy (全体) | Accuracy (public) | Accuracy (private) | AUC (全体) |
| --- | --- | --- | --- | --- |
| Boosted Decision Tree | **0.8675** | 0.8662 | 0.8679 | **0.8156** |
| 線形回帰 | 0.8524 | 0.8504 | 0.8529 | 0.6210 |

### 補足: Accuracy の解釈に注意

ランクDの割合は約 14.8%（不均衡データ）であり、「全件を非D」と予測するだけで
Accuracy は約 **0.852** になります。そのため 0.5 閾値の Accuracy だけでは
両モデルの差はほとんど現れません。

一方で判別性能を表す **AUC では Boosted Decision Tree (0.81) が線形回帰 (0.68) を大きく上回り**、
非線形な特徴量の相互作用を捉えられている分、実質的な予測力は BDT が優れています。
連続値スコアをそのまま使う／閾値を調整する用途では BDT を推奨します。
