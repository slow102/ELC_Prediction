"""Plot LSTM, Transformer, and Improved-model from saved prediction CSVs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODELS = ["LSTM", "Transformer", "Improved-model"]
COLORS = {
    "LSTM": "#FF7F0E",
    "Transformer": "#2CA02C",
    "Improved-model": "#D62728",
}
GROUND_TRUTH_COLOR = "#1F77B4"


def load_prediction(outputs_dir, model, input_len, pred_len, seed):
    path = (
        outputs_dir / model / f"{input_len}_to_{pred_len}"
        / f"seed_{seed}" / "predictions.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction CSV: {path}\n"
            "Train the model or run scripts/export_existing_predictions.py first."
        )
    frame = pd.read_csv(path)
    required = {"sample_id", "forecast_day", "ground_truth", "predicted_power"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def select_curve(frame, mode, sample_id):
    if mode == "mean":
        return frame.groupby("forecast_day", as_index=False)[
            ["ground_truth", "predicted_power"]
        ].mean()
    selected = frame.loc[frame["sample_id"] == sample_id].copy()
    if selected.empty:
        raise ValueError(f"sample_id={sample_id} is not present in the CSV.")
    return selected.sort_values("forecast_day")


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", linewidth=0.65, color="#B0B0B0", alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9CA3AF")


def plot_comparison(args):
    outputs_dir = Path(args.outputs_dir)
    fig, axes = plt.subplots(
        len(args.pred_lens), 1,
        figsize=(14, 5.1 * len(args.pred_lens)),
        squeeze=False, facecolor="white",
    )

    for ax, pred_len in zip(axes[:, 0], args.pred_lens):
        frames = {
            model: load_prediction(
                outputs_dir, model, args.input_len, pred_len, args.seed,
            )
            for model in MODELS
        }
        curves = {
            model: select_curve(frame, args.mode, args.sample_id)
            for model, frame in frames.items()
        }
        reference = curves[MODELS[0]]["ground_truth"].to_numpy()
        days = curves[MODELS[0]]["forecast_day"].to_numpy()
        for model in MODELS[1:]:
            candidate = curves[model]["ground_truth"].to_numpy()
            np.testing.assert_allclose(reference, candidate, rtol=1e-5, atol=1e-3)

        style_axis(ax)
        ax.plot(
            days, reference, color=GROUND_TRUTH_COLOR, linewidth=1.5,
            label="Ground Truth", zorder=5,
        )
        for model in MODELS:
            prediction = curves[model]["predicted_power"].to_numpy()
            mae = np.mean(np.abs(prediction - reference))
            ax.plot(
                days, prediction, color=COLORS[model], linewidth=1.3,
                alpha=0.95, label=f"{model} (MAE {mae:,.1f})",
            )
        descriptor = "mean test curve" if args.mode == "mean" else f"test sample {args.sample_id}"
        ax.set_title(
            f"Power Forecast Comparison | {args.input_len} -> {pred_len}",
            loc="left", fontsize=14, fontweight="bold", pad=27,
        )
        ax.text(
            0, 1.005, f"{descriptor} | seed {args.seed} | original target scale",
            transform=ax.transAxes, fontsize=9.5, color="#6B7280", va="bottom",
        )
        ax.set_xlabel("Forecast Day")
        ax.set_ylabel("Daily Power (original scale)")
        ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=9)

    fig.suptitle(
        "LSTM vs Transformer vs Improved-model",
        fontsize=16, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] saved comparison figure: {save_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare three models using their saved prediction CSV files."
    )
    parser.add_argument("--outputs_dir", default=str(ROOT / "outputs"))
    parser.add_argument("--input_len", type=int, default=90)
    parser.add_argument("--pred_lens", type=int, nargs="+", default=[90, 365])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", choices=["sample", "mean"], default="sample")
    parser.add_argument("--sample_id", type=int, default=0)
    parser.add_argument(
        "--save_path", default=str(ROOT / "outputs" / "three_model_comparison.png"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    plot_comparison(parse_args())
