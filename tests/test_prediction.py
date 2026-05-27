from __future__ import annotations

import unittest

from src import artifacts, config
from src.prediction import (
    available_batteries,
    available_models,
    cycle_summary,
    derived_rul_from_soh_curve,
    filter_predictions,
)


ARTIFACTS_AVAILABLE = config.ARTIFACTS_DIR.exists()


@unittest.skipUnless(ARTIFACTS_AVAILABLE, "Local artifacts directory is not available")
class PredictionHelperTests(unittest.TestCase):
    def test_filter_predictions_returns_clean_benchmark_rows(self) -> None:
        predictions = artifacts.load_baseline_predictions()
        models = available_models(predictions, scenario="clean_benchmark")
        self.assertIn("ExtraTrees", models)
        batteries = available_batteries(predictions, "clean_benchmark", "ExtraTrees")
        self.assertGreater(len(batteries), 0)

        rows = filter_predictions(
            predictions,
            scenario="clean_benchmark",
            model="ExtraTrees",
            battery_id=batteries[0],
        )
        self.assertGreater(len(rows), 0)
        self.assertEqual(rows.iloc[0]["battery_id"], batteries[0])

    def test_filter_predictions_raises_for_missing_model(self) -> None:
        predictions = artifacts.load_baseline_predictions()
        with self.assertRaises(ValueError):
            filter_predictions(
                predictions,
                scenario="clean_benchmark",
                model="MissingModel",
            )

    def test_cycle_summary_for_rul(self) -> None:
        predictions = artifacts.load_baseline_predictions()
        rows = filter_predictions(predictions, "clean_benchmark", "ExtraTrees")
        summary = cycle_summary(rows, cycle_index=int(rows.iloc[0]["cycle_index"]), task="rul")
        self.assertEqual(summary["target"], "rul_cycles")
        self.assertIn("predicted_value", summary)
        self.assertIn("absolute_error", summary)

    def test_soh_derived_rul_summary(self) -> None:
        predictions = artifacts.load_soh_predictions()
        models = available_models(predictions, scenario="clean_benchmark")
        self.assertIn("RandomForest", models)
        batteries = available_batteries(predictions, "clean_benchmark", "RandomForest")
        rows = filter_predictions(predictions, "clean_benchmark", "RandomForest", batteries[0])
        summary = cycle_summary(rows, cycle_index=int(rows.iloc[0]["cycle_index"]), task="soh")
        derived = derived_rul_from_soh_curve(rows, summary["cycle_index"], threshold=0.80)
        self.assertEqual(summary["target"], "soh")
        self.assertIn(derived["status"], {"threshold_reached", "threshold_not_reached_in_available_curve"})


if __name__ == "__main__":
    unittest.main()
