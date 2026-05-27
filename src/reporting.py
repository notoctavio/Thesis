"""Reusable result summaries for the thesis app and documentation."""

from __future__ import annotations

import pandas as pd

from .config import SCENARIO_LABELS, SCENARIO_NOTES


def drop_naive_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Remove naive baselines from model-ranking tables."""

    return metrics.loc[~metrics["model"].str.contains("Naive", case=False, na=False)].copy()


def best_models_by_scenario(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the lowest-RMSE model row per scenario."""

    clean = drop_naive_models(metrics)
    test_rows = clean.loc[clean["split"].eq("test")].copy()
    if test_rows.empty:
        return test_rows
    idx = test_rows.groupby("scenario")["RMSE"].idxmin()
    return test_rows.loc[idx].sort_values("scenario").reset_index(drop=True)


def best_model_name(metrics: pd.DataFrame, scenario: str) -> str | None:
    """Return the lowest-RMSE non-naive model name for one scenario."""

    rows = drop_naive_models(metrics)
    rows = rows.loc[rows["scenario"].eq(scenario)]
    if "split" in rows.columns:
        rows = rows.loc[rows["split"].eq("test")]
    if rows.empty:
        return None
    return str(rows.sort_values("RMSE").iloc[0]["model"])


def metric_row(metrics: pd.DataFrame, scenario: str, model: str) -> pd.Series:
    """Return a single metric row for the selected scenario and model."""

    rows = metrics.loc[(metrics["scenario"] == scenario) & (metrics["model"] == model)]
    if rows.empty:
        raise ValueError(f"No metrics found for scenario={scenario!r}, model={model!r}")
    return rows.iloc[0]


def scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


def scenario_note(scenario: str) -> str:
    return SCENARIO_NOTES.get(scenario, "No scenario note is documented.")


def format_metric_row(row: pd.Series) -> dict[str, float | str]:
    """Convert a metric row into display-friendly values."""

    return {
        "scenario": str(row["scenario"]),
        "model": str(row["model"]),
        "RMSE": round(float(row["RMSE"]), 4),
        "MAE": round(float(row["MAE"]), 4),
        "R2": round(float(row["R2"]), 4),
    }
