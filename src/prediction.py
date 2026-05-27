"""Prediction-table helpers used by notebooks, tests, and the Streamlit app."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _sorted_unique(series: pd.Series) -> list[Any]:
    values = [value for value in series.dropna().unique().tolist()]
    return sorted(values)


def available_models(predictions: pd.DataFrame, scenario: str | None = None) -> list[str]:
    frame = predictions
    if scenario is not None:
        frame = frame.loc[frame["scenario"] == scenario]
    return _sorted_unique(frame["model"])


def available_batteries(
    predictions: pd.DataFrame,
    scenario: str,
    model: str | None = None,
) -> list[str]:
    frame = predictions.loc[predictions["scenario"] == scenario]
    if model is not None:
        frame = frame.loc[frame["model"] == model]
    return _sorted_unique(frame["battery_id"])


def filter_predictions(
    predictions: pd.DataFrame,
    scenario: str,
    model: str,
    battery_id: str | None = None,
) -> pd.DataFrame:
    """Filter prediction rows and fail loudly when the selection is empty."""

    required_cols = {"scenario", "model", "battery_id", "cycle_index"}
    missing_cols = required_cols - set(predictions.columns)
    if missing_cols:
        raise ValueError(f"Prediction table misses required columns: {sorted(missing_cols)}")

    frame = predictions.loc[
        (predictions["scenario"] == scenario) & (predictions["model"] == model)
    ].copy()
    if battery_id is not None:
        frame = frame.loc[frame["battery_id"] == battery_id].copy()

    if frame.empty:
        details = f"scenario={scenario!r}, model={model!r}, battery_id={battery_id!r}"
        raise ValueError(f"No predictions found for {details}")

    return frame.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)


def nearest_cycle_row(predictions: pd.DataFrame, cycle_index: int) -> pd.Series:
    """Return the row whose cycle index is closest to the requested cycle."""

    if predictions.empty:
        raise ValueError("Cannot select a cycle from an empty prediction table.")

    distances = (predictions["cycle_index"] - int(cycle_index)).abs()
    return predictions.loc[distances.idxmin()]


def cycle_summary(
    predictions: pd.DataFrame,
    cycle_index: int,
    task: str,
) -> dict[str, Any]:
    """Summarize the prediction at the selected cycle for direct RUL or SOH."""

    row = nearest_cycle_row(predictions, cycle_index)
    common = {
        "battery_id": row.get("battery_id"),
        "cycle_index": int(row.get("cycle_index")),
        "capacity_ah_clean": float(row.get("capacity_ah_clean"))
        if pd.notna(row.get("capacity_ah_clean"))
        else None,
    }

    if task == "rul":
        return {
            **common,
            "target": "rul_cycles",
            "true_value": float(row.get("rul_cycles")),
            "predicted_value": float(row.get("prediction")),
            "absolute_error": float(row.get("abs_error")),
        }

    if task == "soh":
        return {
            **common,
            "target": "soh",
            "true_value": float(row.get("soh")),
            "predicted_value": float(row.get("pred_soh")),
            "absolute_error": float(row.get("abs_soh_error")),
            "true_rul_cycles": float(row.get("rul_cycles"))
            if pd.notna(row.get("rul_cycles"))
            else None,
        }

    raise ValueError("task must be either 'rul' or 'soh'")


def derived_rul_from_soh_curve(
    predictions: pd.DataFrame,
    current_cycle: int,
    threshold: float,
    prediction_col: str = "pred_soh",
) -> dict[str, Any]:
    """Estimate remaining cycles until predicted SOH crosses the threshold."""

    if prediction_col not in predictions.columns:
        raise ValueError(f"Missing SOH prediction column: {prediction_col}")

    curve = predictions.sort_values("cycle_index").copy()
    current_cycle = int(current_cycle)
    future = curve.loc[curve["cycle_index"] >= current_cycle]
    crossed = future.loc[future[prediction_col] <= threshold]

    if crossed.empty:
        return {
            "eol_cycle": None,
            "remaining_cycles": None,
            "threshold": float(threshold),
            "status": "threshold_not_reached_in_available_curve",
        }

    eol_cycle = int(crossed.iloc[0]["cycle_index"])
    return {
        "eol_cycle": eol_cycle,
        "remaining_cycles": max(eol_cycle - current_cycle, 0),
        "threshold": float(threshold),
        "status": "threshold_reached",
    }
