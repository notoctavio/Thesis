from __future__ import annotations

import unittest

from src import artifacts, config
from src.reporting import best_models_by_scenario


ARTIFACTS_AVAILABLE = config.ARTIFACTS_DIR.exists()


@unittest.skipUnless(ARTIFACTS_AVAILABLE, "Local artifacts directory is not available")
class ArtifactLoadingTests(unittest.TestCase):
    def test_required_artifacts_exist(self) -> None:
        paths = artifacts.validate_required_artifacts()
        self.assertGreaterEqual(len(paths), 6)
        for path in paths:
            self.assertTrue(path.exists(), path)

    def test_scenarios_include_expected_benchmarks(self) -> None:
        scenarios = artifacts.available_scenarios()
        self.assertIn("all_eligible", scenarios)
        self.assertIn("clean_benchmark", scenarios)
        self.assertIn("nasa_classic_4", scenarios)

    def test_model_comparison_has_best_rows(self) -> None:
        metrics = artifacts.load_model_comparison()
        best = best_models_by_scenario(metrics)
        self.assertGreaterEqual(len(best), 3)
        self.assertTrue({"scenario", "model", "RMSE", "MAE", "R2"}.issubset(best.columns))

    def test_soh_combined_metrics_include_sequence_when_available(self) -> None:
        metrics = artifacts.load_soh_model_comparison()
        self.assertIn("clean_benchmark", metrics["scenario"].unique())
        if "family" in metrics.columns:
            self.assertIn("baseline", metrics["family"].unique())
            self.assertIn("sequence", metrics["family"].unique())


if __name__ == "__main__":
    unittest.main()
