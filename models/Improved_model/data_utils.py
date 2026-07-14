"""Leakage-free data construction for Improved-model.

The only extra future information is deterministic calendar information and a
seasonal baseline fitted on the training target-date block.  Validation labels
are used solely to choose the recent-residual decay hyperparameter; test labels
are never inspected during construction.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils.data_utils import (
    PowerDataset,
    add_time_features,
    check_required_columns,
    create_windows_for_target_range,
    get_default_feature_cols,
    get_future_time_feature_cols,
    get_target_date_blocks,
    load_daily_data,
    scale_dataset,
)


@dataclass
class SeasonalBaselineConfig:
    harmonics: int
    ridge_alpha: float
    residual_history_days: int
    residual_tau: float
    validation_mse_original: float


def _calendar_design(index: pd.DatetimeIndex, harmonics: int) -> np.ndarray:
    """Long-term calendar basis known for every prediction date."""
    index = pd.DatetimeIndex(index)
    day = (index - index[0]).days.to_numpy(dtype=np.float64)
    doy = index.dayofyear.to_numpy(dtype=np.float64)
    dow = index.dayofweek.to_numpy(dtype=np.float64)

    columns = [day / max(float(day.max()), 1.0)]
    for k in range(1, harmonics + 1):
        columns.extend([
            np.sin(2.0 * np.pi * k * doy / 365.2425),
            np.cos(2.0 * np.pi * k * doy / 365.2425),
        ])
    columns.extend([
        np.sin(2.0 * np.pi * dow / 7.0),
        np.cos(2.0 * np.pi * dow / 7.0),
        (dow >= 5).astype(np.float64),
    ])
    return np.column_stack(columns)


def _known_future_features(
    index: pd.DatetimeIndex,
    target_starts: np.ndarray,
    pred_len: int,
    harmonics: int,
) -> np.ndarray:
    """Annual Fourier basis and normalized forecast distance."""
    index = pd.DatetimeIndex(index)
    doy = index.dayofyear.to_numpy(dtype=np.float64)
    dow = index.dayofweek.to_numpy(dtype=np.float64)
    global_time = (
        (index - index[0]).days.to_numpy(dtype=np.float64)
        / max(float((index[-1] - index[0]).days), 1.0)
    )

    features = []
    for start in target_starts:
        sl = slice(start, start + pred_len)
        cols = []
        for k in range(1, harmonics + 1):
            cols.extend([
                np.sin(2.0 * np.pi * k * doy[sl] / 365.2425),
                np.cos(2.0 * np.pi * k * doy[sl] / 365.2425),
            ])
        cols.extend([
            np.sin(2.0 * np.pi * dow[sl] / 7.0),
            np.cos(2.0 * np.pi * dow[sl] / 7.0),
            (dow[sl] >= 5).astype(np.float64),
            global_time[sl],
            np.arange(1, pred_len + 1, dtype=np.float64) / pred_len,
            np.log1p(np.arange(1, pred_len + 1, dtype=np.float64))
            / np.log1p(pred_len),
        ])
        features.append(np.column_stack(cols))
    return np.asarray(features, dtype=np.float32)


def _build_baselines(
    y_all: np.ndarray,
    calendar_prediction: np.ndarray,
    target_starts: np.ndarray,
    pred_len: int,
    history_days: int,
    tau: float,
) -> np.ndarray:
    horizon = np.arange(pred_len, dtype=np.float64)
    decay = np.exp(-horizon / tau)
    baselines = []
    for start in target_starts:
        first = max(0, start - history_days)
        recent_residual = np.mean(
            y_all[first:start] - calendar_prediction[first:start]
        )
        baselines.append(
            calendar_prediction[start:start + pred_len]
            + recent_residual * decay
        )
    return np.asarray(baselines, dtype=np.float32)


def build_hast_dataloaders(
    csv_path: str,
    input_len: int = 90,
    pred_len: int = 90,
    batch_size: int = 32,
    date_col: str = "date",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    long_val_days: int = 400,
    long_test_days: int = 400,
    harmonics: int = 8,
    ridge_alpha: float = 100.0,
    recency_half_life: float = 180.0,
):
    df = add_time_features(load_daily_data(csv_path, date_col=date_col))
    target_col = "global_active_power"
    feature_cols = get_default_feature_cols(df)
    base_future_cols = get_future_time_feature_cols()
    check_required_columns(df, feature_cols, target_col)

    blocks = get_target_date_blocks(
        len(df), input_len, pred_len, train_ratio, val_ratio,
        long_val_days, long_test_days,
    )
    arrays = {}
    starts = {}
    for split, (block_start, block_end) in blocks.items():
        x, _, y, split_starts = create_windows_for_target_range(
            df, feature_cols, base_future_cols, target_col,
            block_start, block_end, input_len, pred_len,
        )
        arrays[split] = [x, y]
        starts[split] = split_starts

    # Identical train-only scaling policy to the LSTM/Transformer baselines.
    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["val"]
    x_test, y_test = arrays["test"]
    (
        x_train_s, y_train_s, x_val_s, y_val_s, x_test_s, y_test_s,
        x_scaler, y_scaler,
    ) = scale_dataset(x_train, y_train, x_val, y_val, x_test, y_test)
    scaled = {
        "train": (x_train_s, y_train_s),
        "val": (x_val_s, y_val_s),
        "test": (x_test_s, y_test_s),
    }

    # Fit the explicit annual/weekly component on training dates only.
    calendar_x = _calendar_design(df.index, harmonics)
    y_all = df[target_col].to_numpy(dtype=np.float64)
    train_end = blocks["train"][1]
    seasonal_ridge = Ridge(alpha=ridge_alpha)
    seasonal_ridge.fit(calendar_x[:train_end], y_all[:train_end])
    calendar_prediction = seasonal_ridge.predict(calendar_x)

    # Validation-only choice of how quickly the most recent level correction
    # should disappear.  This is hyperparameter selection, not test leakage.
    best = None
    for history_days in (7, 14, 30, 60, 90):
        for tau in (7.0, 14.0, 30.0, 60.0, 90.0):
            val_base = _build_baselines(
                y_all, calendar_prediction, starts["val"], pred_len,
                history_days, tau,
            )
            score = mean_squared_error(y_val.reshape(-1), val_base.reshape(-1))
            if best is None or score < best[0]:
                best = (score, history_days, tau)
    val_score, history_days, tau = best
    baseline_config = SeasonalBaselineConfig(
        harmonics=harmonics,
        ridge_alpha=ridge_alpha,
        residual_history_days=history_days,
        residual_tau=tau,
        validation_mse_original=float(val_score),
    )

    future_feature_cols = (
        [
            name
            for k in range(1, harmonics + 1)
            for name in (f"annual_sin_{k}", f"annual_cos_{k}")
        ]
        + [
            "weekly_sin", "weekly_cos", "is_weekend", "global_time",
            "horizon_fraction", "log_horizon_fraction",
            "seasonal_baseline_scaled",
        ]
    )

    loaders = {}
    baselines_original = {}
    for split in ("train", "val", "test"):
        future = _known_future_features(
            df.index, starts[split], pred_len, harmonics,
        )
        baseline = _build_baselines(
            y_all, calendar_prediction, starts[split], pred_len,
            history_days, tau,
        )
        baseline_scaled = (
            (baseline - float(y_scaler.mean_[0]))
            / float(y_scaler.scale_[0])
        )
        future = np.concatenate(
            [future, baseline_scaled[:, :, None].astype(np.float32)], axis=-1,
        )
        dataset = PowerDataset(scaled[split][0], future, scaled[split][1])
        sampler = None
        shuffle = split == "train"
        if split == "train" and recency_half_life > 0:
            age = np.arange(len(dataset) - 1, -1, -1, dtype=np.float64)
            weights = np.exp(-np.log(2.0) * age / recency_half_life)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
        )
        baselines_original[split] = baseline

    print(
        f"  [data] records={len(df)} | features={len(feature_cols)} | "
        + "windows train/val/test="
        + "/".join(str(len(starts[name])) for name in ("train", "val", "test"))
        + f" | seasonal history={history_days}, tau={tau:g}"
    )

    return (
        loaders["train"], loaders["val"], loaders["test"],
        x_scaler, y_scaler, feature_cols, future_feature_cols, df,
        baseline_config, baselines_original, starts,
    )
