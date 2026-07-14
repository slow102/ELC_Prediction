"""Regenerate model figures directly from standardized prediction CSVs."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_utils import plot_mean_prediction, plot_one_prediction


MODELS = ["LSTM", "Transformer", "Improved-model"]


def csv_to_arrays(path):
    frame = pd.read_csv(path)
    prediction = frame.pivot(
        index="sample_id", columns="forecast_day", values="predicted_power",
    ).sort_index().sort_index(axis=1)
    ground_truth = frame.pivot(
        index="sample_id", columns="forecast_day", values="ground_truth",
    ).sort_index().sort_index(axis=1)
    if prediction.isna().any().any() or ground_truth.isna().any().any():
        raise ValueError(f"Incomplete sample/horizon grid in {path}")
    return prediction.to_numpy(), ground_truth.to_numpy()


def main():
    count = 0
    for model_name in MODELS:
        model_root = ROOT / "outputs" / model_name
        for path in sorted(model_root.glob("*_to_*/seed_*/predictions.csv")):
            preds, targets = csv_to_arrays(path)
            folder = path.parent
            input_len = int(path.parents[1].name.split("_to_")[0])
            pred_len = preds.shape[1]
            plot_one_prediction(
                preds, targets, folder / "power_forecast_sample.png", 0,
                f"{model_name} Power Forecast | {input_len} -> {pred_len}",
                model_name=model_name,
            )
            plot_mean_prediction(
                preds, targets, folder / "power_forecast_mean.png",
                f"{model_name} Mean Power Forecast | {input_len} -> {pred_len}",
                model_name=model_name,
            )
            count += 1
    print(f"[plot] regenerated figures from {count} prediction CSV files")


if __name__ == "__main__":
    main()
