from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class PowerDataset(Dataset):
    """
    PyTorch Dataset for daily household power forecasting.

    X shape: [num_samples, input_len, num_features]
    future_time shape: [num_samples, pred_len, num_future_features]
    y shape: [num_samples, pred_len]
    """
    def __init__(
        self,
        X: np.ndarray,
        future_time: np.ndarray,
        y: np.ndarray,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.future_time = torch.tensor(future_time, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.future_time[idx], self.y[idx]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column names to lower-case English names.
    This helps handle columns such as '日期' or 'Date'.
    """
    df = df.copy()

    rename_map = {
        "日期": "date",
        "Date": "date",
        "datetime": "date",
        "Datetime": "date",

        "Global_active_power": "global_active_power",
        "Global_reactive_power": "global_reactive_power",
        "Voltage": "voltage",
        "Global_intensity": "global_intensity",
        "Sub_metering_1": "sub_metering_1",
        "Sub_metering_2": "sub_metering_2",
        "Sub_metering_3": "sub_metering_3",
        "Sub_metering_remainder": "sub_metering_remainder",

        "global active power": "global_active_power",
        "global reactive power": "global_reactive_power",
        "sub metering 1": "sub_metering_1",
        "sub metering 2": "sub_metering_2",
        "sub metering 3": "sub_metering_3",
        "sub metering remainder": "sub_metering_remainder",
    }

    df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
    df.columns = [str(c).strip() for c in df.columns]

    return df


def load_daily_data(csv_path: str, date_col: str = "date") -> pd.DataFrame:
    """
    Load processed daily CSV.
    The CSV should contain one row per day and all missing values should already be handled.
    """
    csv_file = Path(csv_path).expanduser()
    if not csv_file.is_absolute() and not csv_file.exists():
        project_relative = Path(__file__).resolve().parents[1] / csv_file
        if project_relative.exists():
            csv_file = project_relative

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Cannot find daily dataset: {csv_file.resolve()}\n"
            "Pass --csv_path with an absolute path, or run the preprocessing "
            "notebook first."
        )

    df = pd.read_csv(csv_file)
    df = normalize_columns(df)

    if date_col not in df.columns:
        if "date" in df.columns:
            date_col = "date"
        else:
            raise ValueError(
                f"Cannot find date column '{date_col}'. "
                f"Available columns are: {list(df.columns)}"
            )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily calendar features.
    Household power consumption usually has weekly and seasonal patterns.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    month = df.index.month
    dayofyear = df.index.dayofyear
    weekday = df.index.weekday

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365)

    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    df["is_weekend"] = (weekday >= 5).astype(int)

    return df


def get_default_feature_cols(df: pd.DataFrame):
    """
    Return the default feature columns used by the LSTM model.
    Only columns that actually exist in the dataframe are returned.
    """
    candidate_cols = [
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
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
    ]

    return [col for col in candidate_cols if col in df.columns]


def get_future_time_feature_cols():
    """Calendar variables known in advance for every forecast date."""
    return [
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
    ]


def check_required_columns(df: pd.DataFrame, feature_cols, target_col: str):
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    nan_count = df[feature_cols + [target_col]].isna().sum()
    if nan_count.sum() > 0:
        print("Columns with missing values:")
        print(nan_count[nan_count > 0])
        raise ValueError("Please handle missing values before building the dataset.")


def create_windows_for_target_range(
    df: pd.DataFrame,
    feature_cols,
    future_time_cols,
    target_col: str,
    target_start_idx: int,
    target_end_idx: int,
    input_len: int = 90,
    pred_len: int = 90,
):
    """
    Build windows whose complete prediction target lies in one date block.

    For each sample:
        X = past input_len days of all feature columns
        y = next pred_len days of target_col

    target_start_idx is inclusive and target_end_idx is exclusive.
    Input history may come from the preceding block, but target dates may not
    cross the block boundary.
    """
    feature_data = df[feature_cols].values.astype(np.float32)
    future_time_data = df[future_time_cols].values.astype(np.float32)
    target_data = df[target_col].values.astype(np.float32)

    X, future_time, y = [], [], []
    target_starts = []

    first_target_start = max(input_len, target_start_idx)
    last_target_start = min(
        target_end_idx - pred_len,
        len(df) - pred_len,
    )

    if last_target_start < first_target_start:
        raise ValueError(
            "Target block is too short to build a complete sample: "
            f"block=[{target_start_idx}, {target_end_idx}), "
            f"input_len={input_len}, pred_len={pred_len}."
        )

    for target_start in range(first_target_start, last_target_start + 1):
        input_start = target_start - input_len
        target_end = target_start + pred_len

        X.append(feature_data[input_start:target_start])
        future_time.append(future_time_data[target_start:target_end])
        y.append(target_data[target_start:target_end])
        target_starts.append(target_start)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(future_time, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(target_starts, dtype=np.int64),
    )


def get_target_date_blocks(
    num_days: int,
    input_len: int,
    pred_len: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    long_val_days: int = 400,
    long_test_days: int = 400,
):
    """
    Return non-overlapping target-date blocks.

    For the 365-day task, reserve 400 target days for validation and 400
    target days for testing. With 1,440 daily records this gives
    640 / 400 / 400 days.

    For shorter tasks, use chronological ratio-based date blocks.
    """
    if pred_len == 365:
        train_end = num_days - long_val_days - long_test_days
        val_end = num_days - long_test_days
    else:
        train_end = int(round(num_days * train_ratio))
        val_end = int(round(num_days * (train_ratio + val_ratio)))

    if not 0 < train_end < val_end < num_days:
        raise ValueError(
            "Invalid target-date split: "
            f"num_days={num_days}, train_end={train_end}, val_end={val_end}."
        )

    blocks = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, num_days),
    }

    for name, (start, end) in blocks.items():
        available_target_days = end - max(
            start,
            input_len if name == "train" else start,
        )
        if available_target_days < pred_len:
            raise ValueError(
                f"{name} target block is too short for pred_len={pred_len}: "
                f"[{start}, {end})."
            )

    return blocks


def scale_dataset(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Fit scalers only on the training set to avoid information leakage.
    X and y use separate StandardScaler objects.
    """
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    num_features = X_train.shape[-1]

    X_train_2d = X_train.reshape(-1, num_features)
    X_val_2d = X_val.reshape(-1, num_features)
    X_test_2d = X_test.reshape(-1, num_features)

    X_train_scaled = x_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test_2d).reshape(X_test.shape)

    y_train_2d = y_train.reshape(-1, 1)
    y_val_2d = y_val.reshape(-1, 1)
    y_test_2d = y_test.reshape(-1, 1)

    y_train_scaled = y_scaler.fit_transform(y_train_2d).reshape(y_train.shape)
    y_val_scaled = y_scaler.transform(y_val_2d).reshape(y_val.shape)
    y_test_scaled = y_scaler.transform(y_test_2d).reshape(y_test.shape)

    return (
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        X_test_scaled, y_test_scaled,
        x_scaler, y_scaler,
    )


def build_dataloaders(
    csv_path: str,
    input_len: int = 90,
    pred_len: int = 90,
    batch_size: int = 32,
    date_col: str = "date",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    long_val_days: int = 400,
    long_test_days: int = 400,
):
    """
    Main dataset-building function.
    """
    df = load_daily_data(csv_path, date_col=date_col)
    df = add_time_features(df)

    target_col = "global_active_power"
    feature_cols = get_default_feature_cols(df)
    future_time_cols = get_future_time_feature_cols()

    check_required_columns(df, feature_cols, target_col)
    check_required_columns(df, future_time_cols, target_col)

    blocks = get_target_date_blocks(
        num_days=len(df),
        input_len=input_len,
        pred_len=pred_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        long_val_days=long_val_days,
        long_test_days=long_test_days,
    )

    datasets = {}
    target_starts = {}
    for split_name, (target_start_idx, target_end_idx) in blocks.items():
        X_split, future_time_split, y_split, starts_split = (
            create_windows_for_target_range(
                df=df,
                feature_cols=feature_cols,
                future_time_cols=future_time_cols,
                target_col=target_col,
                target_start_idx=target_start_idx,
                target_end_idx=target_end_idx,
                input_len=input_len,
                pred_len=pred_len,
            )
        )
        datasets[split_name] = (X_split, future_time_split, y_split)
        target_starts[split_name] = starts_split

    X_train, future_train, y_train = datasets["train"]
    X_val, future_val, y_val = datasets["val"]
    X_test, future_test, y_test = datasets["test"]

    train_target_end = target_starts["train"][-1] + pred_len - 1
    val_target_start = target_starts["val"][0]
    val_target_end = target_starts["val"][-1] + pred_len - 1
    test_target_start = target_starts["test"][0]

    assert train_target_end < val_target_start
    assert val_target_end < test_target_start

    print(
        f"  [data] records={len(df)} | features={len(feature_cols)} | "
        f"windows train/val/test={len(X_train)}/{len(X_val)}/{len(X_test)}"
    )

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        x_scaler, y_scaler,
    ) = scale_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    train_loader = DataLoader(
        PowerDataset(X_train, future_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PowerDataset(X_val, future_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        PowerDataset(X_test, future_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        x_scaler,
        y_scaler,
        feature_cols,
        future_time_cols,
        df,
    )
