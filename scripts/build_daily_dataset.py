"""Build daily electricity forecasting datasets from the UCI minute data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_POWER_TXT = ROOT / "data" / "raw" / "household_power_consumption.txt"
RAW_POWER_GZ = ROOT / "data" / "raw" / "household_power_consumption.txt.gz"
DAILY_WEATHER = (
    ROOT
    / "data"
    / "intermediate"
    / "weather_daily_repeated_2006-12-16_2010-11-26.csv"
)
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
FINAL_DIR = ROOT / "data" / "final"

MIN_COVERAGE = 0.80

RAW_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
SUM_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
AVERAGE_COLUMNS = ["Voltage", "Global_intensity"]
FINAL_COLUMNS = [
    "date",
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
]


def read_minute_data() -> tuple[pd.DataFrame, dict[str, object]]:
    raw_power = RAW_POWER_TXT if RAW_POWER_TXT.exists() else RAW_POWER_GZ
    if not raw_power.exists():
        raise FileNotFoundError(
            "Missing raw power data. Expected either "
            f"{RAW_POWER_TXT} or {RAW_POWER_GZ}."
        )
    data = pd.read_csv(
        raw_power,
        sep=";",
        na_values=["?", "nan", "NaN", ""],
        low_memory=False,
    )
    data["datetime"] = pd.to_datetime(
        data["Date"] + " " + data["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    data = data.sort_values("datetime").set_index("datetime")

    for column in RAW_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    duplicate_timestamps = int(data.index.duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(f"Found {duplicate_timestamps} duplicate timestamps.")

    full_index = pd.date_range(data.index.min(), data.index.max(), freq="1min")
    missing_timestamps = int(len(full_index.difference(data.index)))
    data = data.reindex(full_index)
    data.index.name = "datetime"

    missing_any = data[RAW_COLUMNS].isna().any(axis=1)
    missing_all = data[RAW_COLUMNS].isna().all(axis=1)
    diagnostics = {
        "raw_rows": int(len(full_index) - missing_timestamps),
        "theoretical_minute_rows": int(len(full_index)),
        "missing_timestamps_added": missing_timestamps,
        "duplicate_timestamps": duplicate_timestamps,
        "rows_with_any_numeric_missing": int(missing_any.sum()),
        "rows_with_all_numeric_missing": int(missing_all.sum()),
        "missing_by_column": {
            key: int(value)
            for key, value in data[RAW_COLUMNS].isna().sum().items()
        },
        "minute_start": str(data.index.min()),
        "minute_end": str(data.index.max()),
    }
    return data, diagnostics


def aggregate_daily(
    minute_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    valid_minutes = (
        minute_data["Global_active_power"].notna().resample("D").sum().astype(int)
    )
    total_minutes = (
        minute_data["Global_active_power"].resample("D").size().astype(int)
    )
    coverage = valid_minutes / total_minutes

    daily_sum = minute_data[SUM_COLUMNS].resample("D").sum(min_count=1)
    corrected_sum = daily_sum.div(valid_minutes.replace(0, np.nan), axis=0).mul(
        total_minutes, axis=0
    )
    daily_average = minute_data[AVERAGE_COLUMNS].resample("D").mean()

    low_quality = coverage < MIN_COVERAGE
    corrected_sum.loc[low_quality, :] = np.nan
    daily_average.loc[low_quality, :] = np.nan

    before_imputation = pd.concat([corrected_sum, daily_average], axis=1)
    before_imputation["valid_minutes"] = valid_minutes
    before_imputation["total_minutes"] = total_minutes
    before_imputation["coverage"] = coverage
    before_imputation["missing_ratio"] = 1 - coverage
    before_imputation["low_quality_flag"] = low_quality.astype(int)

    full_day_mask = total_minutes == 1440
    before_imputation = before_imputation.loc[full_day_mask].copy()

    daily = before_imputation.copy()
    feature_columns = SUM_COLUMNS + AVERAGE_COLUMNS
    originally_missing = daily[feature_columns].isna().any(axis=1)
    daily["_month"] = daily.index.month
    daily["_weekday"] = daily.index.weekday

    median_filled = pd.Series(False, index=daily.index)
    for column in feature_columns:
        was_missing = daily[column].isna()
        group_median = daily.groupby(["_month", "_weekday"])[column].transform(
            "median"
        )
        daily[column] = daily[column].fillna(group_median)
        median_filled |= was_missing & daily[column].notna()

    remaining_before_interpolation = daily[feature_columns].isna().any(axis=1)
    daily[feature_columns] = (
        daily[feature_columns].interpolate(method="time").ffill().bfill()
    )
    interpolation_filled = remaining_before_interpolation & daily[
        feature_columns
    ].notna().all(axis=1)

    daily = daily.drop(columns=["_month", "_weekday"])
    daily["imputed_flag"] = originally_missing.astype(int)
    daily["imputation_method"] = "none"
    daily.loc[median_filled, "imputation_method"] = "month_weekday_median"
    daily.loc[interpolation_filled, "imputation_method"] = "time_interpolation"

    daily["Sub_metering_remainder"] = (
        daily["Global_active_power"] * 1000 / 60
        - daily[
            ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        ].sum(axis=1)
    ).clip(lower=0)

    daily = daily.rename(
        columns={
            "Global_active_power": "global_active_power",
            "Global_reactive_power": "global_reactive_power",
            "Voltage": "voltage",
            "Global_intensity": "global_intensity",
            "Sub_metering_1": "sub_metering_1",
            "Sub_metering_2": "sub_metering_2",
            "Sub_metering_3": "sub_metering_3",
            "Sub_metering_remainder": "sub_metering_remainder",
        }
    )
    before_imputation = before_imputation.rename(
        columns={
            "Global_active_power": "global_active_power",
            "Global_reactive_power": "global_reactive_power",
            "Voltage": "voltage",
            "Global_intensity": "global_intensity",
            "Sub_metering_1": "sub_metering_1",
            "Sub_metering_2": "sub_metering_2",
            "Sub_metering_3": "sub_metering_3",
        }
    )

    diagnostics = {
        "calendar_days_before_removing_partial_ends": int(len(total_minutes)),
        "full_days_retained": int(full_day_mask.sum()),
        "partial_days_removed": int((~full_day_mask).sum()),
        "days_with_any_missing_minutes": int(
            ((1 - coverage.loc[full_day_mask]) > 0).sum()
        ),
        "fully_missing_days": int(
            (valid_minutes.loc[full_day_mask] == 0).sum()
        ),
        "low_quality_days_below_80_percent": int(
            low_quality.loc[full_day_mask].sum()
        ),
        "days_filled_by_month_weekday_median": int(median_filled.sum()),
        "days_filled_by_time_interpolation": int(interpolation_filled.sum()),
        "daily_start": daily.index.min().strftime("%Y-%m-%d"),
        "daily_end": daily.index.max().strftime("%Y-%m-%d"),
    }
    return before_imputation, daily, diagnostics


def merge_weather(daily: pd.DataFrame) -> pd.DataFrame:
    weather = pd.read_csv(DAILY_WEATHER, parse_dates=["Date"])
    weather = weather.drop(columns=["AAAAMM"])
    weather = weather.rename(columns={"Date": "date"}).set_index("date")
    merged = daily.join(weather, how="left", validate="1:1")
    weather_columns = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
    if merged[weather_columns].isna().any().any():
        raise ValueError("Weather merge introduced missing values.")

    merged.insert(0, "date", merged.index.strftime("%Y-%m-%d"))
    return merged.reset_index(drop=True)[FINAL_COLUMNS]


def save_outputs(
    final_dataset: pd.DataFrame,
    diagnostics: dict[str, object],
) -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final_dataset.to_csv(
        FINAL_DIR / "daily_household_power_weather.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics["final_dataset"] = {
        "rows": int(len(final_dataset)),
        "columns": int(final_dataset.shape[1]),
        "missing_cells": int(final_dataset.isna().sum().sum()),
        "date_start": final_dataset["date"].iloc[0],
        "date_end": final_dataset["date"].iloc[-1],
        "negative_counts": {
            column: int((final_dataset[column] < 0).sum())
            for column in [
                "global_active_power",
                "global_reactive_power",
                "global_intensity",
                "sub_metering_1",
                "sub_metering_2",
                "sub_metering_3",
                "sub_metering_remainder",
            ]
        },
        "voltage_min": float(final_dataset["voltage"].min()),
        "voltage_max": float(final_dataset["voltage"].max()),
    }


def main() -> None:
    minute_data, minute_diagnostics = read_minute_data()
    before_imputation, daily_power, daily_diagnostics = aggregate_daily(
        minute_data
    )
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    before_imputation.to_csv(
        INTERMEDIATE_DIR / "daily_power_before_imputation.csv",
        index=True,
        index_label="date",
        encoding="utf-8-sig",
    )
    daily_power.to_csv(
        INTERMEDIATE_DIR / "daily_household_power_processed.csv",
        index=True,
        index_label="date",
        encoding="utf-8-sig",
    )
    final_dataset = merge_weather(daily_power)

    feature_columns = [
        "global_active_power",
        "global_reactive_power",
        "voltage",
        "global_intensity",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3",
        "sub_metering_remainder",
    ]
    if final_dataset[feature_columns].isna().any().any():
        raise ValueError("Final electricity features still contain missing values.")
    if (daily_power["coverage"] < 0).any() or (
        daily_power["coverage"] > 1
    ).any():
        raise ValueError("Coverage is outside [0, 1].")

    diagnostics = {
        "configuration": {
            "minimum_daily_coverage": MIN_COVERAGE,
            "daily_sum_columns": SUM_COLUMNS,
            "daily_average_columns": AVERAGE_COLUMNS,
            "imputation_order": [
                "same_month_and_weekday_median",
                "time_interpolation",
                "forward_fill",
                "backward_fill",
            ],
            "weather_station": "PARIS-MONTSOURIS (75114001)",
        },
        "minute_diagnostics": minute_diagnostics,
        "daily_diagnostics": daily_diagnostics,
    }
    save_outputs(
        final_dataset,
        diagnostics,
    )
    print(
        "Saved intermediate daily tables and final model-ready dataset. "
        f"Final rows={len(final_dataset)}, columns={final_dataset.shape[1]}"
    )


if __name__ == "__main__":
    main()
