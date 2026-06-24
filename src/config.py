"""Central configuration for the Li-ion battery RUL thesis project."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
EXAMPLES_DIR = DATA_DIR / "examples"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATA_SPLITS_DIR = DATA_DIR / "splits"

ARTIFACTS_DIR = ROOT / "artifacts"
FEATURES_DIR = PROCESSED_DATA_DIR
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
SPLITS_DIR = DATA_SPLITS_DIR
FIGURES_DIR = ARTIFACTS_DIR / "figures"

FEATURES_PATH = FEATURES_DIR / "battery_cycle_features_v2.csv"
BASELINE_FEATURE_COLUMNS_PATH = FEATURES_DIR / "baseline_feature_columns_v2.json"
SOH_FEATURE_COLUMNS_PATH = FEATURES_DIR / "soh_feature_columns_v2.json"
SCENARIOS_PATH = SPLITS_DIR / "modeling_scenarios_v1.json"
EXAMPLE_INFERENCE_CSV_PATH = EXAMPLES_DIR / "b0007_soh_inference_example.csv"
MODEL_MANIFEST_PATH = MODELS_DIR / "model_manifest.json"

BASELINE_PREDICTIONS_PATH = PREDICTIONS_DIR / "baseline_test_predictions.csv"
SEQUENCE_PREDICTIONS_PATH = PREDICTIONS_DIR / "sequence_test_predictions.csv"
SOH_PREDICTIONS_PATH = PREDICTIONS_DIR / "soh_test_predictions.csv"
SOH_SEQUENCE_PREDICTIONS_PATH = PREDICTIONS_DIR / "soh_sequence_test_predictions.csv"
SOH_ALL_PREDICTIONS_PATH = PREDICTIONS_DIR / "soh_all_test_predictions.csv"

ALL_MODEL_TEST_COMPARISON_PATH = METRICS_DIR / "all_model_test_comparison.csv"
BASELINE_TEST_METRICS_PATH = METRICS_DIR / "baseline_test_metrics.csv"
SEQUENCE_TEST_METRICS_PATH = METRICS_DIR / "sequence_test_metrics.csv"
SOH_TEST_METRICS_PATH = METRICS_DIR / "soh_test_metrics.csv"
SOH_SEQUENCE_TEST_METRICS_PATH = METRICS_DIR / "soh_sequence_test_metrics.csv"
SOH_ALL_MODEL_TEST_COMPARISON_PATH = METRICS_DIR / "soh_all_model_test_comparison.csv"
SOH_DERIVED_RUL_TO_80_METRICS_PATH = METRICS_DIR / "soh_derived_rul_to_80_metrics.csv"

SCENARIO_LABELS = {
    "all_eligible": "Toate bateriile eligibile",
    "clean_benchmark": "Clean benchmark",
    "nasa_classic_4": "NASA clasic B0005/B0006/B0007/B0018",
}

SCENARIO_NOTES = {
    "all_eligible": (
        "Benchmark strict de stres, folosind toate bateriile cu suficiente cicluri "
        "de descărcare. Este mai dificil și scoate în evidență eterogenitatea datasetului."
    ),
    "clean_benchmark": (
        "Scenariul principal de raportare, folosind baterii cu suficiente cicluri "
        "și degradare coerentă a capacității/SOH."
    ),
    "nasa_classic_4": (
        "Subsetul NASA clasic folosit în multe articole și repository-uri publice. "
        "Este util pentru comparație, dar nu reprezintă întregul dataset eterogen."
    ),
}

DEFAULT_EOL_THRESHOLDS = {
    "all_eligible": 0.80,
    "clean_benchmark": 0.80,
    "nasa_classic_4": 0.70,
}

DIRECT_RUL_TASK = "RUL direct (benchmark)"
SOH_TASK = "RUL derivat din SOH"
