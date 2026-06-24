"""CSV inference helpers for the public Streamlit demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import artifacts, config
from .prediction import derived_rul_from_soh_curve


SOH_LAG_FEATURES = [
    "prev_soh",
    "prev_soh_delta1",
    "prev_soh_rollmean5",
    "prev_soh_rollstd5",
    "prev_soh_slope5",
]


@dataclass(frozen=True)
class InferenceResult:
    """Structured result returned by SOH CSV inference."""

    model_id: str
    model_display_name: str
    scenario: str
    threshold: float
    battery_id: str
    predictions: pd.DataFrame
    eol_cycle: int | None
    remaining_cycles_at_first_cycle: int | None
    status: str


def load_model_manifest() -> dict[str, Any]:
    """Load the public model manifest."""

    return artifacts.load_json(config.MODEL_MANIFEST_PATH)


def available_model_ids(manifest: dict[str, Any] | None = None) -> list[str]:
    """Return public inference model ids in manifest order."""

    manifest = manifest or load_model_manifest()
    return list(manifest.get("models", {}).keys())


def get_model_info(model_id: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return manifest metadata for one public inference model."""

    manifest = manifest or load_model_manifest()
    models = manifest.get("models", {})
    if model_id not in models:
        available = ", ".join(models.keys())
        raise KeyError(f"Unknown model_id {model_id!r}. Available models: {available}")
    return models[model_id]


def load_soh_feature_columns() -> list[str]:
    """Return the SOH feature columns expected by public models."""

    feature_config = artifacts.load_json(config.SOH_FEATURE_COLUMNS_PATH)
    return list(feature_config["feature_cols"])


def missing_feature_columns(frame: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Return feature columns missing from an uploaded frame."""

    return [col for col in feature_cols if col not in frame.columns]


def add_soh_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add SOH history features when ground-truth SOH is available."""

    if "soh" not in frame.columns:
        return frame.copy()

    out = _sort_for_curve(frame.copy())
    groups = out.groupby("battery_id", sort=False) if "battery_id" in out.columns else [(None, out)]

    for _, group in groups:
        idx = group.sort_values("cycle_index").index if "cycle_index" in group.columns else group.index
        soh = out.loc[idx, "soh"].astype(float)
        prev_soh = soh.shift(1).fillna(1.0)
        out.loc[idx, "prev_soh"] = prev_soh
        out.loc[idx, "prev_soh_delta1"] = soh.diff().shift(1).fillna(0.0)
        out.loc[idx, "prev_soh_rollmean5"] = prev_soh.rolling(5, min_periods=1).mean().fillna(1.0)
        out.loc[idx, "prev_soh_rollstd5"] = prev_soh.rolling(5, min_periods=2).std().fillna(0.0)
        out.loc[idx, "prev_soh_slope5"] = (
            prev_soh.rolling(5, min_periods=3)
            .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 3 else 0.0, raw=False)
            .fillna(0.0)
        )

    return out


def validate_feature_columns(frame: pd.DataFrame, feature_cols: list[str]) -> None:
    """Raise a clear error when an uploaded CSV is not model-ready."""

    missing = missing_feature_columns(frame, feature_cols)
    if missing:
        sample = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" (+{len(missing) - 8} altele)"
        raise ValueError(f"Lipsesc coloane necesare pentru model: {sample}{suffix}")


def load_public_model(model_id: str) -> Any:
    """Load a public joblib model by manifest id."""

    model_info = get_model_info(model_id)
    return artifacts.load_joblib(config.ROOT / model_info["path"])


def load_example_frame() -> pd.DataFrame:
    """Load the public B0007 CSV example used by the Streamlit app."""

    return artifacts.load_csv(config.EXAMPLE_INFERENCE_CSV_PATH)


def _display_battery_id(frame: pd.DataFrame) -> str:
    if "battery_id" not in frame.columns:
        return "CSV încărcat"
    values = sorted(str(value) for value in frame["battery_id"].dropna().unique())
    if not values:
        return "CSV încărcat"
    if len(values) == 1:
        return values[0]
    return f"{len(values)} baterii"


def _sort_for_curve(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ["battery_id", "cycle_index"] if col in frame.columns]
    if not sort_cols:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_cols).reset_index(drop=True)


def predict_soh_from_frame(
    frame: pd.DataFrame,
    model_id: str,
    threshold: float | None = None,
) -> InferenceResult:
    """Predict SOH for an uploaded model-ready cycle-level CSV."""

    model_info = get_model_info(model_id)
    feature_cols = load_soh_feature_columns()
    curve = add_soh_lag_features(frame)
    validate_feature_columns(curve, feature_cols)

    curve = _sort_for_curve(curve)
    model = load_public_model(model_id)
    pred_soh = np.asarray(model.predict(curve[feature_cols]), dtype=float)
    curve["pred_soh"] = np.clip(pred_soh, 0.0, 1.30)

    if "soh" in curve.columns:
        curve["abs_soh_error"] = (curve["soh"].astype(float) - curve["pred_soh"]).abs()

    selected_threshold = float(
        threshold if threshold is not None else model_info.get("default_threshold", 0.80)
    )
    current_cycle = int(curve["cycle_index"].min())
    derived = derived_rul_from_soh_curve(
        curve,
        current_cycle=current_cycle,
        threshold=selected_threshold,
        prediction_col="pred_soh",
    )

    curve["eol_threshold"] = selected_threshold
    curve["model_id"] = model_id
    curve["scenario"] = model_info["scenario"]

    return InferenceResult(
        model_id=model_id,
        model_display_name=model_info["display_name"],
        scenario=model_info["scenario"],
        threshold=selected_threshold,
        battery_id=_display_battery_id(curve),
        predictions=curve,
        eol_cycle=derived["eol_cycle"],
        remaining_cycles_at_first_cycle=derived["remaining_cycles"],
        status=derived["status"],
    )
