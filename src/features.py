"""Feature engineering for the rank-D prediction task.

Each product belongs to a manufacturing batch identified by ``(line, batch_count)``.
The machine log stores 20 time-step sensor readings per batch, so batch-level
sensor statistics are joined onto every product together with its own
positional attributes (tray / position).
"""

from __future__ import annotations

import pandas as pd

SENSOR_COLUMNS = ["temperature_1", "temperature_2", "temperature_3", "pressure"]


def build_batch_features(machine_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-time-step machine log into one row per batch."""
    grouped = machine_log.groupby(["line", "batch_count"])

    agg: dict[str, list[str]] = {col: ["mean", "std", "min", "max"] for col in SENSOR_COLUMNS}
    agg["maintenance_count"] = ["max"]
    features = grouped.agg(agg)
    features.columns = [f"{col}_{stat}" for col, stat in features.columns]

    # Trend within a batch (last reading minus first reading) captures drift.
    ordered = machine_log.sort_values("process_time")
    first = ordered.groupby(["line", "batch_count"])[SENSOR_COLUMNS].first()
    last = ordered.groupby(["line", "batch_count"])[SENSOR_COLUMNS].last()
    for col in SENSOR_COLUMNS:
        features[f"{col}_trend"] = last[col] - first[col]
        features[f"{col}_range"] = features[f"{col}_max"] - features[f"{col}_min"]

    return features.reset_index()


def build_features(products: pd.DataFrame, batch_features: pd.DataFrame) -> pd.DataFrame:
    """Join batch-level sensor features onto product-level rows."""
    merged = products.merge(batch_features, on=["line", "batch_count"], how="left")
    return merged


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model input columns (everything except identifiers/target).

    ``batch_count`` is excluded: it is a monotonically increasing batch index
    (train uses low values, test starts at ~10000) so feeding it as a raw
    numeric would force the models to extrapolate to out-of-distribution values
    at test time. It is still used to *join* sensor features, just not as input.
    """
    exclude = {"product_id", "rank", "is_rank_d", "batch_count"}
    return [c for c in df.columns if c not in exclude]
