"""Train and evaluate the proposed Improved-model."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.Improved_model.data_utils import build_hast_dataloaders
from models.Improved_model.models import ImprovedModel
from utils.plot_utils import (
    plot_mean_prediction,
    plot_one_prediction,
    save_per_horizon_metrics,
    save_prediction_csv,
)
from utils.train_utils import (
    inverse_transform_y,
    predict,
    train_model,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def original_metrics(targets, preds):
    mse = mean_squared_error(targets.reshape(-1), preds.reshape(-1))
    mae = mean_absolute_error(targets.reshape(-1), preds.reshape(-1))
    return float(mse), float(mae), float(np.sqrt(mse))


def run_one(args, pred_len, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recency_half_life = args.recency_half_life
    if recency_half_life <= 0:
        # Selected from validation behavior: the 90-day task has 829 training
        # windows and benefits from stronger recency weighting, whereas the
        # 365-day task has only 186 windows and needs nearly the full sample.
        recency_half_life = 30.0 if pred_len <= 90 else 180.0
    (
        train_loader, val_loader, test_loader, x_scaler, y_scaler,
        feature_cols, future_cols, df, baseline_config,
        baselines_original, target_starts,
    ) = build_hast_dataloaders(
        csv_path=args.csv_path,
        input_len=args.input_len,
        pred_len=pred_len,
        batch_size=args.batch_size,
        date_col=args.date_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        long_val_days=args.long_val_days,
        long_test_days=args.long_test_days,
        harmonics=args.harmonics,
        ridge_alpha=args.ridge_alpha,
        recency_half_life=recency_half_life,
    )

    model = ImprovedModel(
        input_size=len(feature_cols),
        future_feature_size=len(future_cols),
        pred_len=pred_len,
        d_model=args.d_model,
        d_time=args.d_time,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    num_parameters = count_parameters(model)
    print(
        f"  [run] seed={seed} | device={device} | params={num_parameters:,} | "
        f"recency_half_life={recency_half_life:g}"
    )

    save_dir = (
        Path(args.output_dir)
        / f"{args.input_len}_to_{pred_len}"
        / f"seed_{seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "Improved-model",
        "architecture": "ImprovedModel",
        "input_len": args.input_len,
        "pred_len": pred_len,
        "feature_cols": feature_cols,
        "future_feature_cols": future_cols,
        "seasonal_baseline": asdict(baseline_config),
        "d_model": args.d_model,
        "d_time": args.d_time,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "loss": args.loss,
        "recency_half_life": recency_half_life,
        "num_parameters": num_parameters,
        "seed": seed,
    }
    with (save_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    model, best_val_loss = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        weight_decay=args.weight_decay,
        save_path=str(save_dir / "best_model.pth"),
        optimizer_name="adamw",
        loss_name=args.loss,
        huber_delta=args.huber_delta,
        grad_clip=args.grad_clip,
        use_lr_scheduler=True,
        include_initial_state=True,
    )
    # Calibrate the correction amplitude on validation data.  This one-scalar
    # convex projection is the MSE-optimal blend between the deterministic
    # seasonal baseline (alpha=0) and the neural correction (alpha=1).  It is
    # fitted without using any test labels and is especially useful for the
    # small 365-day training set.
    val_preds_scaled, val_targets_scaled = predict(model, val_loader, device)
    baseline_val_scaled = (
        (baselines_original["val"] - float(y_scaler.mean_[0]))
        / float(y_scaler.scale_[0])
    )
    val_correction = val_preds_scaled - baseline_val_scaled
    val_residual = val_targets_scaled - baseline_val_scaled
    denominator = float(np.sum(val_correction ** 2))
    if denominator <= 1e-12:
        correction_alpha = 0.0
    else:
        correction_alpha = float(np.clip(
            np.sum(val_correction * val_residual) / denominator,
            0.0,
            1.0,
        ))
    calibrated_val_scaled = (
        baseline_val_scaled + correction_alpha * val_correction
    )
    calibrated_val_mse = mean_squared_error(
        val_targets_scaled.reshape(-1), calibrated_val_scaled.reshape(-1),
    )
    with (save_dir / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "correction_alpha": correction_alpha,
                "calibrated_val_mse_standardized": float(calibrated_val_mse),
                "selection_data": "validation only",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    test_preds_scaled, test_targets_scaled = predict(model, test_loader, device)
    baseline_test_scaled = (
        (baselines_original["test"] - float(y_scaler.mean_[0]))
        / float(y_scaler.scale_[0])
    )
    preds_scaled = baseline_test_scaled + correction_alpha * (
        test_preds_scaled - baseline_test_scaled
    )
    preds = inverse_transform_y(preds_scaled, y_scaler)
    targets = inverse_transform_y(test_targets_scaled, y_scaler)
    mse_normalized, mae_normalized, _ = original_metrics(
        test_targets_scaled, preds_scaled,
    )
    mse_original, mae_original, rmse_original = original_metrics(targets, preds)
    baseline_preds = baselines_original["test"]
    baseline_mse, baseline_mae, baseline_rmse = original_metrics(
        targets, baseline_preds,
    )

    np.save(save_dir / "preds.npy", preds)
    np.save(save_dir / "targets.npy", targets)
    np.save(save_dir / "seasonal_baseline_preds.npy", baseline_preds)
    save_prediction_csv(
        preds, targets, save_dir / "predictions.csv", "Improved-model",
        args.input_len, pred_len, seed,
    )
    save_per_horizon_metrics(
        preds, targets, save_dir / "per_horizon_metrics.csv",
    )
    plot_one_prediction(
        preds, targets, save_dir / "power_forecast_sample.png", 0,
        f"Improved-model Power Forecast | {args.input_len} -> {pred_len}",
        model_name="Improved-model",
    )
    plot_mean_prediction(
        preds, targets, save_dir / "power_forecast_mean.png",
        f"Improved-model Mean Power Forecast | {args.input_len} -> {pred_len}",
        model_name="Improved-model",
    )

    correction = preds - baseline_preds
    return {
        "pred_len": pred_len,
        "seed": seed,
        "best_val_loss": best_val_loss,
        "mse_normalized": mse_normalized,
        "mae_normalized": mae_normalized,
        "mse_original": mse_original,
        "mae_original": mae_original,
        "rmse_original": rmse_original,
        "baseline_mse_original": baseline_mse,
        "baseline_mae_original": baseline_mae,
        "baseline_rmse_original": baseline_rmse,
        "mean_abs_neural_correction": float(np.mean(np.abs(correction))),
        "correction_alpha": correction_alpha,
        "calibrated_val_mse": calibrated_val_mse,
        "num_parameters": num_parameters,
        "residual_history_days": baseline_config.residual_history_days,
        "residual_tau": baseline_config.residual_tau,
    }


def run_task(args, pred_len, seeds):
    records = []
    print(f"\n[Improved-model] {args.input_len}->{pred_len} | seeds={list(seeds)}")
    for seed in seeds:
        result = run_one(args, pred_len, seed)
        records.append(result)
        print(
            f"  [test] seed={seed} | MSE={result['mse_original']:,.2f} | "
            f"MAE={result['mae_original']:,.2f} | "
            f"RMSE={result['rmse_original']:,.2f}"
        )

    result_df = pd.DataFrame(records)
    task_dir = Path(args.output_dir) / f"{args.input_len}_to_{pred_len}"
    task_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(task_dir / "all_seed_results.csv", index=False)
    summary = {"pred_len": pred_len}
    for metric in (
        "mse_normalized", "mae_normalized", "mse_original",
        "mae_original", "rmse_original", "mean_abs_neural_correction",
        "correction_alpha", "calibrated_val_mse",
    ):
        summary[f"{metric}_mean"] = result_df[metric].mean()
        summary[f"{metric}_std"] = result_df[metric].std(ddof=0)
    pd.DataFrame([summary]).to_csv(task_dir / "summary_results.csv", index=False)

    print(
        f"  [summary] MSE={summary['mse_original_mean']:,.2f}+/-"
        f"{summary['mse_original_std']:,.2f} | "
        f"MAE={summary['mae_original_mean']:,.2f}+/-"
        f"{summary['mae_original_std']:,.2f} | "
        f"RMSE={summary['rmse_original_mean']:,.2f}+/-"
        f"{summary['rmse_original_std']:,.2f}"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train proposed Improved-model.")
    parser.add_argument(
        "--csv_path", type=str,
        default=str(PROJECT_ROOT / "data" / "final" /
                    "daily_household_power_weather.csv"),
    )
    parser.add_argument("--date_col", type=str, default="date")
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "outputs" / "Improved-model"),
    )
    parser.add_argument("--input_len", type=int, default=90)
    parser.add_argument("--pred_lens", type=int, nargs="+", default=[90, 365])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--d_time", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--dim_feedforward", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--harmonics", type=int, default=8)
    parser.add_argument("--ridge_alpha", type=float, default=100.0)
    parser.add_argument(
        "--recency_half_life", type=float, default=-1.0,
        help=(
            "Half-life in training windows for recency-weighted sampling. "
            "The default auto-selects 30 for <=90 days and 180 for 365 days."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=["mse", "mae", "huber"], default="mse")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--long_val_days", type=int, default=400)
    parser.add_argument("--long_test_days", type=int, default=400)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[2026, 2027, 2028, 2029, 2030],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = [run_task(args, p, args.seeds) for p in args.pred_lens]
    pd.DataFrame(summaries).to_csv(
        Path(args.output_dir) / "all_task_summary.csv", index=False,
    )
    print(f"\n[Improved-model] finished | outputs={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
