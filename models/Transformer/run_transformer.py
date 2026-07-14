import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import build_dataloaders
from utils.plot_utils import (
    plot_mean_prediction,
    plot_one_prediction,
    save_per_horizon_metrics,
    save_prediction_csv,
)
from utils.train_utils import (
    predict_and_evaluate,
    train_model,
)
from models.Transformer.models.transformer_model import (
    PowerPatchTransformer,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one_experiment(args, pred_len: int, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        train_loader,
        val_loader,
        test_loader,
        x_scaler,
        y_scaler,
        feature_cols,
        future_time_cols,
        df,
    ) = build_dataloaders(
        csv_path=args.csv_path,
        input_len=args.input_len,
        pred_len=pred_len,
        batch_size=args.batch_size,
        date_col=args.date_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        long_val_days=args.long_val_days,
        long_test_days=args.long_test_days,
    )

    target_col = "global_active_power"
    target_feature_index = feature_cols.index(target_col)
    use_weekly_residual = args.use_weekly_residual

    model = PowerPatchTransformer(
        input_size=len(feature_cols),
        future_feature_size=len(future_time_cols),
        pred_len=pred_len,
        target_feature_index=target_feature_index,
        target_x_mean=float(x_scaler.mean_[target_feature_index]),
        target_x_scale=float(x_scaler.scale_[target_feature_index]),
        target_y_mean=float(y_scaler.mean_[0]),
        target_y_scale=float(y_scaler.scale_[0]),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        use_weekly_residual=use_weekly_residual,
        use_revin=args.use_revin,
    ).to(device)

    num_parameters = count_trainable_parameters(model)
    print(
        f"  [run] seed={seed} | device={device} | params={num_parameters:,} | "
        f"weekly_residual={use_weekly_residual} | revin={args.use_revin}"
    )

    save_dir = (
        Path(args.output_dir)
        / f"{args.input_len}_to_{pred_len}"
        / f"seed_{seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "model": "PowerPatchTransformer",
        "input_len": args.input_len,
        "pred_len": pred_len,
        "input_size": len(feature_cols),
        "future_feature_size": len(future_time_cols),
        "feature_cols": feature_cols,
        "future_time_cols": future_time_cols,
        "target_col": target_col,
        "target_feature_index": target_feature_index,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "patch_len": args.patch_len,
        "patch_stride": args.patch_stride,
        "use_weekly_residual": use_weekly_residual,
        "use_revin": args.use_revin,
        "loss": args.loss,
        "num_parameters": num_parameters,
        "seed": seed,
    }
    with (save_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    model, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
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
    )

    eval_result = predict_and_evaluate(
        model=model,
        test_loader=test_loader,
        y_scaler=y_scaler,
        device=device,
    )
    preds = eval_result["preds"]
    targets = eval_result["targets"]

    np.save(save_dir / "preds.npy", preds)
    np.save(save_dir / "targets.npy", targets)
    save_prediction_csv(
        preds, targets, save_dir / "predictions.csv", "Transformer",
        args.input_len, pred_len, seed,
    )
    save_per_horizon_metrics(
        preds=preds,
        targets=targets,
        save_path=save_dir / "per_horizon_metrics.csv",
    )

    plot_one_prediction(
        preds=preds,
        targets=targets,
        save_path=save_dir / "power_forecast_sample.png",
        sample_idx=0,
        title=(
            f"Transformer Power Forecast | {args.input_len} -> {pred_len}"
        ),
        model_name="Transformer",
    )
    plot_mean_prediction(
        preds=preds,
        targets=targets,
        save_path=save_dir / "power_forecast_mean.png",
        title=(
            f"Transformer Mean Power Forecast | {args.input_len} -> {pred_len}"
        ),
        model_name="Transformer",
    )

    return {
        "pred_len": pred_len,
        "seed": seed,
        "best_val_loss": best_val_loss,
        "mse_normalized": eval_result["mse_normalized"],
        "mae_normalized": eval_result["mae_normalized"],
        "mse_original": eval_result["mse_original"],
        "mae_original": eval_result["mae_original"],
        "rmse_original": eval_result["rmse_original"],
        "num_parameters": num_parameters,
        "weekly_residual": use_weekly_residual,
        "num_features": len(feature_cols),
        "feature_cols": ",".join(feature_cols),
        "num_future_features": len(future_time_cols),
        "future_time_cols": ",".join(future_time_cols),
    }


def run_task(args, pred_len: int, seeds):
    records = []

    print(f"\n[Transformer] {args.input_len}->{pred_len} | seeds={list(seeds)}")

    for seed in seeds:
        result = run_one_experiment(args, pred_len=pred_len, seed=seed)
        records.append(result)

        print(
            f"  [test] seed={seed} | MSE={result['mse_original']:,.2f} | "
            f"MAE={result['mae_original']:,.2f} | "
            f"RMSE={result['rmse_original']:,.2f}"
        )

    result_df = pd.DataFrame(records)
    task_dir = (
        Path(args.output_dir)
        / f"{args.input_len}_to_{pred_len}"
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(task_dir / "all_seed_results.csv", index=False)

    summary = {
        "pred_len": pred_len,
        "mse_normalized_mean": result_df["mse_normalized"].mean(),
        "mse_normalized_std": result_df["mse_normalized"].std(ddof=0),
        "mae_normalized_mean": result_df["mae_normalized"].mean(),
        "mae_normalized_std": result_df["mae_normalized"].std(ddof=0),
        "mse_original_mean": result_df["mse_original"].mean(),
        "mse_original_std": result_df["mse_original"].std(ddof=0),
        "mae_original_mean": result_df["mae_original"].mean(),
        "mae_original_std": result_df["mae_original"].std(ddof=0),
        "rmse_original_mean": result_df["rmse_original"].mean(),
        "rmse_original_std": result_df["rmse_original"].std(ddof=0),
    }
    pd.DataFrame([summary]).to_csv(
        task_dir / "summary_results.csv",
        index=False,
    )

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
    parser = argparse.ArgumentParser(
        description=(
            "Weekly-patch Transformer for daily household power forecasting."
        )
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data"
            / "final"
            / "daily_household_power_weather.csv"
        ),
    )
    parser.add_argument("--date_col", type=str, default="date")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "Transformer"),
    )

    parser.add_argument("--input_len", type=int, default=90)
    parser.add_argument("--pred_lens", type=int, nargs="+", default=[90, 365])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patch_len", type=int, default=7)
    parser.add_argument("--patch_stride", type=int, default=7)
    parser.add_argument(
        "--use_weekly_residual",
        action="store_true",
        help=(
            "Predict corrections to a repeated last-week baseline. "
            "The default is direct forecasting because it validated better "
            "on this dataset."
        ),
    )
    parser.add_argument(
        "--use_revin",
        action="store_true",
        help=(
            "Enable per-window history normalization and target-scale "
            "restoration as an ablation setting."
        ),
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "mae", "huber"],
        default="mse",
    )
    parser.add_argument("--huber_delta", type=float, default=1.0)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--long_val_days", type=int, default=400)
    parser.add_argument("--long_test_days", type=int, default=400)

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028, 2029, 2030],
    )

    return parser.parse_args()


def main():
    args = parse_args()

    all_summaries = []
    for pred_len in args.pred_lens:
        summary = run_task(args, pred_len=pred_len, seeds=args.seeds)
        all_summaries.append(summary)

    all_summary_df = pd.DataFrame(all_summaries)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_summary_df.to_csv(
        Path(args.output_dir) / "all_task_summary.csv",
        index=False,
    )

    print(f"\n[Transformer] finished | outputs={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
