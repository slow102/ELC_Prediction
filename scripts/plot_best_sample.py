"""Find and plot the best test sample from a prediction CSV."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_utils import plot_one_prediction


def main(args):
    csv_path = Path(args.csv_path)
    frame = pd.read_csv(csv_path)
    metrics = frame.groupby("sample_id", as_index=False).agg(
        mae=("absolute_error", "mean"),
        mse=("error", lambda x: float(np.mean(np.asarray(x) ** 2))),
    )
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics = metrics.sort_values([args.metric, "mae", "rmse"]).reset_index(drop=True)
    best_id = int(metrics.iloc[0]["sample_id"])
    metrics.to_csv(csv_path.parent / "sample_metrics.csv", index=False)

    prediction = frame.pivot(
        index="sample_id", columns="forecast_day", values="predicted_power",
    ).sort_index().sort_index(axis=1)
    ground_truth = frame.pivot(
        index="sample_id", columns="forecast_day", values="ground_truth",
    ).sort_index().sort_index(axis=1)
    position = list(prediction.index).index(best_id)
    model_name = str(frame["model"].iloc[0])
    input_len = int(frame["input_len"].iloc[0])
    pred_len = int(frame["pred_len"].iloc[0])
    save_path = Path(args.save_path) if args.save_path else (
        csv_path.parent / f"power_forecast_best_{args.metric}_sample.png"
    )
    plot_one_prediction(
        prediction.to_numpy(), ground_truth.to_numpy(), save_path,
        sample_idx=position,
        title=f"{model_name} Best-sample Forecast | {input_len} -> {pred_len}",
        model_name=model_name,
        sample_label=f"sample_id={best_id}, selected by minimum {args.metric.upper()}",
    )
    row = metrics.iloc[0]
    print(
        f"[best] sample_id={best_id} | MAE={row['mae']:.2f} | "
        f"RMSE={row['rmse']:.2f} | figure={save_path.resolve()}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--metric", choices=["mae", "rmse"], default="mae")
    parser.add_argument("--save_path")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
