"""Utilities for loading saved thesis artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import config


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when a required local artifact is missing."""


def _require_path(path: Path) -> Path:
    if not path.exists():
        raise ArtifactNotFoundError(f"Missing artifact: {path}")
    return path


@lru_cache(maxsize=None)
def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV artifact with a clear error for missing files."""

    return pd.read_csv(_require_path(Path(path)))


@lru_cache(maxsize=None)
def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON artifact."""

    with _require_path(Path(path)).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=None)
def load_joblib(path: str | Path) -> Any:
    """Load a joblib model artifact."""

    return joblib.load(_require_path(Path(path)))


def load_features() -> pd.DataFrame:
    return load_csv(config.FEATURES_PATH)


def load_scenarios() -> dict[str, Any]:
    return load_json(config.SCENARIOS_PATH)


def load_baseline_predictions() -> pd.DataFrame:
    return load_csv(config.BASELINE_PREDICTIONS_PATH)


def load_sequence_predictions() -> pd.DataFrame:
    return load_csv(config.SEQUENCE_PREDICTIONS_PATH)


def load_soh_predictions() -> pd.DataFrame:
    return load_csv(config.SOH_PREDICTIONS_PATH)


def load_soh_sequence_predictions() -> pd.DataFrame:
    return load_csv(config.SOH_SEQUENCE_PREDICTIONS_PATH)


def load_soh_all_predictions() -> pd.DataFrame:
    if config.SOH_ALL_PREDICTIONS_PATH.exists():
        return load_csv(config.SOH_ALL_PREDICTIONS_PATH)
    return load_soh_predictions().assign(family="baseline")


def load_model_comparison() -> pd.DataFrame:
    return load_csv(config.ALL_MODEL_TEST_COMPARISON_PATH)


def load_baseline_metrics() -> pd.DataFrame:
    return load_csv(config.BASELINE_TEST_METRICS_PATH)


def load_sequence_metrics() -> pd.DataFrame:
    return load_csv(config.SEQUENCE_TEST_METRICS_PATH)


def load_soh_metrics() -> pd.DataFrame:
    return load_csv(config.SOH_TEST_METRICS_PATH)


def load_soh_sequence_metrics() -> pd.DataFrame:
    return load_csv(config.SOH_SEQUENCE_TEST_METRICS_PATH)


def load_soh_model_comparison() -> pd.DataFrame:
    if config.SOH_ALL_MODEL_TEST_COMPARISON_PATH.exists():
        return load_csv(config.SOH_ALL_MODEL_TEST_COMPARISON_PATH)
    return load_soh_metrics().assign(family="baseline")


def load_soh_derived_rul_metrics() -> pd.DataFrame:
    return load_csv(config.SOH_DERIVED_RUL_TO_80_METRICS_PATH)


def available_scenarios() -> list[str]:
    """Return scenarios in the saved split artifact."""

    return list(load_scenarios().keys())


def validate_required_artifacts() -> list[Path]:
    """Return required artifact paths after validating that they exist."""

    required = [
        config.FEATURES_PATH,
        config.SCENARIOS_PATH,
        config.BASELINE_PREDICTIONS_PATH,
        config.SOH_ALL_PREDICTIONS_PATH if config.SOH_ALL_PREDICTIONS_PATH.exists() else config.SOH_PREDICTIONS_PATH,
        config.ALL_MODEL_TEST_COMPARISON_PATH,
        config.SOH_ALL_MODEL_TEST_COMPARISON_PATH
        if config.SOH_ALL_MODEL_TEST_COMPARISON_PATH.exists()
        else config.SOH_TEST_METRICS_PATH,
    ]
    return [_require_path(path) for path in required]
