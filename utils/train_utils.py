import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    grad_clip: Optional[float] = None,
):
    model.train()
    total_loss = 0.0

    for X, future_time, y in train_loader:
        X = X.to(device)
        future_time = future_time.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X, future_time)
        loss = criterion(pred, y)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(train_loader.dataset)


def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, future_time, y in data_loader:
            X = X.to(device)
            future_time = future_time.to(device)
            y = y.to(device)

            pred = model(X, future_time)
            loss = criterion(pred, y)

            total_loss += loss.item() * X.size(0)

    return total_loss / len(data_loader.dataset)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
    weight_decay: float = 0.0,
    save_path: Optional[str] = None,
    optimizer_name: str = "adam",
    loss_name: str = "mse",
    huber_delta: float = 1.0,
    grad_clip: Optional[float] = None,
    use_lr_scheduler: bool = False,
    include_initial_state: bool = False,
    log_every: int = 5,
):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "mae":
        criterion = nn.L1Loss()
    elif loss_name == "huber":
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        raise ValueError(
            f"Unsupported loss_name='{loss_name}'. Choose mse, mae, or huber."
        )

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer_name='{optimizer_name}'. Choose adam or adamw."
        )

    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1e-6,
        )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    # Residual models can be deliberately initialized as a strong deterministic
    # baseline.  Keeping that state as an early-stopping candidate guarantees
    # that optimization cannot silently discard the baseline when every learned
    # correction is worse on validation data.
    if include_initial_state:
        best_val_loss = evaluate_loss(model, val_loader, criterion, device)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        print(f"  [train] initial baseline | val={best_val_loss:.6f}")
        if save_path is not None:
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_path)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip=grad_clip,
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch == 1 or epoch % max(log_every, 1) == 0 or epoch == epochs:
            print(
                f"  [train] epoch {epoch:03d}/{epochs:03d} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"lr={current_lr:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0

            if save_path is not None:
                Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_path)
        else:
            wait += 1

        if wait >= patience:
            print(
                f"  [train] early stop at epoch {epoch} | "
                f"best_val={best_val_loss:.6f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def predict(model, data_loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, future_time, y in data_loader:
            X = X.to(device)
            future_time = future_time.to(device)

            pred = model(X, future_time).cpu().numpy()
            y = y.numpy()

            all_preds.append(pred)
            all_targets.append(y)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return preds, targets


def inverse_transform_y(y_scaled, y_scaler):
    y_inv = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)
    return y_inv


def predict_and_evaluate(model, test_loader, y_scaler, device):
    preds_scaled, targets_scaled = predict(model, test_loader, device)

    # Standardized-scale metrics match the training and validation loss scale.
    mse_normalized = mean_squared_error(
        targets_scaled.reshape(-1),
        preds_scaled.reshape(-1),
    )
    mae_normalized = mean_absolute_error(
        targets_scaled.reshape(-1),
        preds_scaled.reshape(-1),
    )

    # Keep saved arrays and figures in the original target scale so that
    # prediction curves remain physically interpretable.
    preds = inverse_transform_y(preds_scaled, y_scaler)
    targets = inverse_transform_y(targets_scaled, y_scaler)

    # Original-scale metrics are the main practical evaluation metrics.
    # They have the same physical unit as the target variable.
    mse_original = mean_squared_error(
        targets.reshape(-1),
        preds.reshape(-1),
    )
    mae_original = mean_absolute_error(
        targets.reshape(-1),
        preds.reshape(-1),
    )
    rmse_original = float(np.sqrt(mse_original))

    return {
        "mse_normalized": mse_normalized,
        "mae_normalized": mae_normalized,
        "mse_original": mse_original,
        "mae_original": mae_original,
        "rmse_original": rmse_original,
        "preds": preds,
        "targets": targets,
    }
