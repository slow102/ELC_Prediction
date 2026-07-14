"""Standard prediction exports and publication-style power forecast plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUND_TRUTH_COLOR = "#1F77B4"
PREDICTION_COLOR = "#FF7F0E"


def save_prediction_csv(
    preds,
    targets,
    save_path,
    model_name,
    input_len,
    pred_len,
    seed,
):
    """Save every test-window prediction in one tidy, original-scale CSV."""
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if preds.shape != targets.shape or preds.ndim != 2:
        raise ValueError("preds and targets must be equally shaped 2-D arrays.")

    sample_ids = np.repeat(np.arange(preds.shape[0]), preds.shape[1])
    forecast_days = np.tile(np.arange(1, preds.shape[1] + 1), preds.shape[0])
    prediction = preds.reshape(-1)
    ground_truth = targets.reshape(-1)
    error = prediction - ground_truth
    frame = pd.DataFrame({
        "model": model_name,
        "input_len": int(input_len),
        "pred_len": int(pred_len),
        "seed": int(seed),
        "sample_id": sample_ids,
        "forecast_day": forecast_days,
        "ground_truth": ground_truth,
        "predicted_power": prediction,
        "error": error,
        "absolute_error": np.abs(error),
    })
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(save_path, index=False)
    return frame


def save_per_horizon_metrics(preds, targets, save_path):
    errors = np.asarray(preds) - np.asarray(targets)
    pd.DataFrame({
        "forecast_day": np.arange(1, errors.shape[1] + 1),
        "mae_original": np.mean(np.abs(errors), axis=0),
        "rmse_original": np.sqrt(np.mean(errors ** 2, axis=0)),
        "bias_original": np.mean(errors, axis=0),
    }).to_csv(save_path, index=False)


def _style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#B0B0B0", linestyle="-", linewidth=0.65, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9CA3AF")
    ax.tick_params(colors="#374151")


def _plot_curves(true, pred, save_path, title, model_name, subtitle=None):
    steps = np.arange(1, len(true) + 1)
    error = pred - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    fig, ax = plt.subplots(figsize=(12.5, 5.2), facecolor="white")
    _style_axis(ax)
    ax.plot(
        steps, true, color=GROUND_TRUTH_COLOR, linewidth=1.35,
        label="Ground Truth", zorder=3,
    )
    ax.plot(
        steps, pred, color=PREDICTION_COLOR, linewidth=1.35,
        label=f"{model_name} Prediction", zorder=4,
    )
    ax.set_xlabel("Forecast Day", fontsize=10.5)
    ax.set_ylabel("Daily Power (original scale)", fontsize=10.5)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=28)
    if subtitle:
        ax.text(
            0.0, 1.005, subtitle, transform=ax.transAxes,
            fontsize=9.5, color="#6B7280", va="bottom",
        )
    ax.text(
        0.985, 0.965, f"MAE  {mae:,.2f}\nRMSE {rmse:,.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white",
              "edgecolor": "#D1D5DB", "alpha": 0.94},
    )
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.82))
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_one_prediction(
    preds,
    targets,
    save_path,
    sample_idx=0,
    title="Power Forecast vs Ground Truth",
    model_name="Model",
    sample_label=None,
):
    _plot_curves(
        np.asarray(targets)[sample_idx], np.asarray(preds)[sample_idx],
        save_path, title, model_name,
        subtitle=(sample_label or f"Test sample {sample_idx}")
        + " | original target scale",
    )


def plot_mean_prediction(
    preds,
    targets,
    save_path,
    title="Mean Power Forecast vs Ground Truth",
    model_name="Model",
):
    _plot_curves(
        np.asarray(targets).mean(axis=0), np.asarray(preds).mean(axis=0),
        save_path, title, model_name,
        subtitle=f"Mean over {len(preds)} test windows | original target scale",
    )
