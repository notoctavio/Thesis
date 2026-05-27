#!/usr/bin/env python3
"""Train LSTM/CNN-LSTM sequence models for SOH prediction."""

from __future__ import annotations

import json
import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
SEQ_LEN = 20
BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FEATURE_TABLE = ARTIFACTS / "features" / "battery_cycle_features_v2.csv"
SOH_FEATURES = ARTIFACTS / "features" / "soh_feature_columns_v2.json"
SCENARIOS_PATH = ARTIFACTS / "splits" / "modeling_scenarios_v1.json"
MODEL_DIR = ARTIFACTS / "models"
METRICS_DIR = ARTIFACTS / "metrics"
PRED_DIR = ARTIFACTS / "predictions"

for directory in [MODEL_DIR, METRICS_DIR, PRED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

cache_root = Path(tempfile.gettempdir()) / "licenta-rul-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, prev_soh: np.ndarray, true_soh: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.prev_soh = torch.tensor(prev_soh, dtype=torch.float32)
        self.true_soh = torch.tensor(true_soh, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.prev_soh[index], self.true_soh[index]


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.20):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CNNLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, conv_channels: int = 48, hidden_size: int = 64, dropout: float = 0.20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
        )
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :]).squeeze(-1)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_sequences(df: pd.DataFrame, feature_cols: list[str], target_col: str, seq_len: int = SEQ_LEN):
    x_list = []
    y_list = []
    meta_rows = []
    for battery_id, group in df.groupby("battery_id"):
        g = group.sort_values("cycle_index").reset_index(drop=True)
        x_group = g[feature_cols].to_numpy(dtype=np.float32)
        y_group = g[target_col].to_numpy(dtype=np.float32)
        for end_idx in range(len(g)):
            start_idx = max(0, end_idx - seq_len + 1)
            window = x_group[start_idx : end_idx + 1]
            padded = np.zeros((seq_len, x_group.shape[1]), dtype=np.float32)
            padded[-len(window) :] = window
            x_list.append(padded)
            y_list.append(y_group[end_idx])
            meta_rows.append(
                {
                    "battery_id": battery_id,
                    "cycle_index": int(g.loc[end_idx, "cycle_index"]),
                    "start_time": g.loc[end_idx, "start_time"],
                    "soh": float(g.loc[end_idx, "soh"]),
                    "prev_soh": float(g.loc[end_idx, "prev_soh"]),
                    "soh_delta_target": float(g.loc[end_idx, "soh_delta_target"]),
                    "capacity_ah_clean": float(g.loc[end_idx, "capacity_ah_clean"]),
                    "rul_cycles": float(g.loc[end_idx, "rul_cycles"]),
                }
            )
    return np.stack(x_list), np.asarray(y_list, dtype=np.float32), pd.DataFrame(meta_rows)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["battery_id", "cycle_index"]).copy()
    for _, group in out.groupby("battery_id"):
        idx = group.sort_values("cycle_index").index
        soh = out.loc[idx, "soh"].astype(float)
        prev_soh = soh.shift(1).fillna(1.0)
        out.loc[idx, "prev_soh"] = prev_soh
        out.loc[idx, "soh_delta_target"] = soh - prev_soh
        out.loc[idx, "prev_soh_delta1"] = soh.diff().shift(1).fillna(0.0)
        out.loc[idx, "prev_soh_rollmean5"] = prev_soh.rolling(5, min_periods=1).mean().fillna(1.0)
        out.loc[idx, "prev_soh_rollstd5"] = prev_soh.rolling(5, min_periods=2).std().fillna(0.0)
        out.loc[idx, "prev_soh_slope5"] = prev_soh.rolling(5, min_periods=3).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 3 else 0.0,
            raw=False,
        ).fillna(0.0)
    return out


def reconstruct_soh(prev_soh: np.ndarray, pred_delta: np.ndarray) -> np.ndarray:
    """Reconstruct SOH from previous SOH and predicted residual."""

    return np.clip(np.asarray(prev_soh, dtype=float) + np.asarray(pred_delta, dtype=float), 0.0, 1.30)


def predict_loader(model: nn.Module, loader: DataLoader):
    model.eval()
    pred_deltas = []
    true_deltas = []
    prev_soh_values = []
    true_soh_values = []
    criterion = nn.SmoothL1Loss()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb, prev_soh, true_soh in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            prev_soh = prev_soh.to(DEVICE)
            true_soh = true_soh.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            pred_deltas.append(pred.detach().cpu().numpy())
            true_deltas.append(yb.detach().cpu().numpy())
            prev_soh_values.append(prev_soh.detach().cpu().numpy())
            true_soh_values.append(true_soh.detach().cpu().numpy())
            total_loss += float(loss.item()) * len(xb)
    pred_delta = np.concatenate(pred_deltas)
    true_delta = np.concatenate(true_deltas)
    prev_soh_raw = np.concatenate(prev_soh_values)
    true_soh = np.concatenate(true_soh_values)
    pred_soh = reconstruct_soh(prev_soh_raw, pred_delta)
    return total_loss / len(loader.dataset), pred_soh, true_soh, pred_delta, true_delta


def train_one_model(name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
    model = model.to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val_rmse = np.inf
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb, _, _ in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss, train_pred, train_true, _, _ = predict_loader(model, train_loader)
        val_loss, val_pred, val_true, _, _ = predict_loader(model, val_loader)
        train_metrics = regression_metrics(train_true, train_pred)
        val_metrics = regression_metrics(val_true, val_pred)
        history.append(
            {
                "model": name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_RMSE": train_metrics["RMSE"],
                "val_RMSE": val_metrics["RMSE"],
                "train_MAE": train_metrics["MAE"],
                "val_MAE": val_metrics["MAE"],
                "val_R2": val_metrics["R2"],
            }
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"{name} epoch {epoch:03d}: train RMSE={train_metrics['RMSE']:.4f}, val RMSE={val_metrics['RMSE']:.4f}", flush=True)
        if bad_epochs >= PATIENCE:
            print(f"{name}: early stopping at epoch {epoch}; best epoch {best_epoch}, val RMSE={best_val_rmse:.4f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val_RMSE": best_val_rmse}


def main() -> int:
    model_df = pd.read_csv(FEATURE_TABLE, parse_dates=["start_time", "imp_start_time"])
    model_df = add_lag_features(model_df)
    feature_config = json.loads(SOH_FEATURES.read_text(encoding="utf-8"))
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    feature_cols = feature_config["feature_cols"]
    absolute_target_col = feature_config["target_col"]
    target_col = "soh_delta_target"

    histories = []
    validation_rows = []
    test_rows = []
    prediction_frames = []

    for scenario_name in ["all_eligible", "clean_benchmark", "nasa_classic_4"]:
        print("\n" + "=" * 80, flush=True)
        print(f"SOH sequence scenario: {scenario_name}", flush=True)
        split = scenarios[scenario_name]["split"]
        scenario_df = model_df[model_df["battery_id"].isin(scenarios[scenario_name]["battery_ids"])].copy()
        train_df = scenario_df[scenario_df["battery_id"].isin(split["train_batteries"])].copy()
        val_df = scenario_df[scenario_df["battery_id"].isin(split["validation_batteries"])].copy()
        test_df = scenario_df[scenario_df["battery_id"].isin(split["test_batteries"])].copy()

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        scaler.fit(imputer.fit_transform(train_df[feature_cols]))
        scaled_cols = [f"z_{col}" for col in feature_cols]

        def add_scaled(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out[scaled_cols] = scaler.transform(imputer.transform(out[feature_cols]))
            return out

        train_df = add_scaled(train_df)
        val_df = add_scaled(val_df)
        test_df = add_scaled(test_df)

        x_train, y_train, meta_train = build_sequences(train_df, scaled_cols, target_col)
        x_val, y_val, meta_val = build_sequences(val_df, scaled_cols, target_col)
        x_test, y_test, meta_test = build_sequences(test_df, scaled_cols, target_col)
        print(f"Shapes: {x_train.shape} {x_val.shape} {x_test.shape}", flush=True)

        preprocessor_path = MODEL_DIR / f"soh_sequence_{scenario_name}_preprocessor_v1.joblib"
        joblib.dump(
            {
                "imputer": imputer,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "seq_len": SEQ_LEN,
                "target_col": target_col,
                "absolute_target_col": absolute_target_col,
            },
            preprocessor_path,
        )

        generator = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(
            SequenceDataset(
                x_train,
                y_train,
                meta_train["prev_soh"].to_numpy(dtype=np.float32),
                meta_train[absolute_target_col].to_numpy(dtype=np.float32),
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(
            SequenceDataset(
                x_val,
                y_val,
                meta_val["prev_soh"].to_numpy(dtype=np.float32),
                meta_val[absolute_target_col].to_numpy(dtype=np.float32),
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        test_loader = DataLoader(
            SequenceDataset(
                x_test,
                y_test,
                meta_test["prev_soh"].to_numpy(dtype=np.float32),
                meta_test[absolute_target_col].to_numpy(dtype=np.float32),
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        model_specs = {
            "LSTM": LSTMRegressor(n_features=x_train.shape[-1]),
            "CNN_LSTM": CNNLSTMRegressor(n_features=x_train.shape[-1]),
        }

        for model_name, model in model_specs.items():
            full_name = f"{scenario_name}__{model_name}"
            trained, history, summary = train_one_model(full_name, model, train_loader, val_loader)
            history["scenario"] = scenario_name
            history["base_model"] = model_name
            histories.append(history)
            validation_rows.append({"scenario": scenario_name, "model": model_name, **summary})

            _, pred_test, true_test, pred_delta_test, true_delta_test = predict_loader(trained, test_loader)
            test_rows.append({"scenario": scenario_name, "model": model_name, "split": "test", **regression_metrics(true_test, pred_test)})

            pred_df = meta_test.copy()
            pred_df["scenario"] = scenario_name
            pred_df["model"] = model_name
            pred_df["family"] = "sequence"
            pred_df["pred_soh"] = pred_test
            pred_df["pred_soh_delta"] = pred_delta_test
            pred_df["true_soh_delta"] = true_delta_test
            pred_df["soh_error"] = pred_df["pred_soh"] - pred_df["soh"]
            pred_df["abs_soh_error"] = pred_df["soh_error"].abs()
            prediction_frames.append(pred_df)

            model_path = MODEL_DIR / f"soh_sequence_{scenario_name}_{model_name.lower()}_state.pt"
            torch.save(
                {
                    "scenario": scenario_name,
                    "model_name": model_name,
                    "state_dict": trained.state_dict(),
                    "feature_cols": feature_cols,
                    "seq_len": SEQ_LEN,
                    "n_features": x_train.shape[-1],
                },
                model_path,
            )

    history_all = pd.concat(histories, ignore_index=True)
    validation_summary = pd.DataFrame(validation_rows).sort_values(["scenario", "best_val_RMSE"])
    sequence_metrics = pd.DataFrame(test_rows).sort_values(["scenario", "RMSE"])
    sequence_predictions = pd.concat(prediction_frames, ignore_index=True)

    history_all.to_csv(METRICS_DIR / "soh_sequence_training_history.csv", index=False)
    validation_summary.to_csv(METRICS_DIR / "soh_sequence_validation_summary.csv", index=False)
    sequence_metrics.to_csv(METRICS_DIR / "soh_sequence_test_metrics.csv", index=False)
    sequence_predictions.to_csv(PRED_DIR / "soh_sequence_test_predictions.csv", index=False)

    classical_metrics = pd.read_csv(METRICS_DIR / "soh_test_metrics.csv").assign(family="baseline")
    combined_metrics = pd.concat(
        [classical_metrics, sequence_metrics.assign(family="sequence")],
        ignore_index=True,
    ).sort_values(["scenario", "RMSE"])
    combined_metrics.to_csv(METRICS_DIR / "soh_all_model_test_comparison.csv", index=False)

    classical_preds = pd.read_csv(PRED_DIR / "soh_test_predictions.csv").assign(family="baseline")
    combined_predictions = pd.concat([classical_preds, sequence_predictions], ignore_index=True, sort=False)
    combined_predictions.to_csv(PRED_DIR / "soh_all_test_predictions.csv", index=False)

    print(validation_summary.to_string(index=False), flush=True)
    print(sequence_metrics.to_string(index=False), flush=True)
    print(combined_metrics.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
