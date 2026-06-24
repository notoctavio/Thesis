from __future__ import annotations

import unittest

import pandas as pd

from src import artifacts, config


class PublicInferenceArtifactTests(unittest.TestCase):
    def test_public_inference_artifacts_exist(self) -> None:
        self.assertTrue(config.EXAMPLE_INFERENCE_CSV_PATH.exists(), config.EXAMPLE_INFERENCE_CSV_PATH)
        self.assertTrue(config.MODEL_MANIFEST_PATH.exists(), config.MODEL_MANIFEST_PATH)

        manifest = artifacts.load_json(config.MODEL_MANIFEST_PATH)
        self.assertIn("models", manifest)
        self.assertIn("nasa_classic_4_randomforest", manifest["models"])
        self.assertIn("clean_benchmark_randomforest", manifest["models"])
        self.assertIn("all_eligible_histgradientboosting", manifest["models"])

        for model_info in manifest["models"].values():
            model_path = config.ROOT / model_info["path"]
            self.assertTrue(model_path.exists(), model_path)

    def test_b0007_example_is_model_ready(self) -> None:
        feature_config = artifacts.load_json(config.SOH_FEATURE_COLUMNS_PATH)
        feature_cols = feature_config["feature_cols"]
        example = pd.read_csv(config.EXAMPLE_INFERENCE_CSV_PATH)

        self.assertGreater(len(example), 20)
        self.assertEqual(set(example["battery_id"].unique()), {"B0007"})
        self.assertTrue({"battery_id", "cycle_index", "soh", "rul_cycles"}.issubset(example.columns))
        self.assertEqual([], [col for col in feature_cols if col not in example.columns])


class CsvInferenceBehaviorTests(unittest.TestCase):
    def test_missing_feature_columns_are_reported(self) -> None:
        from src import inference

        missing = inference.missing_feature_columns(
            pd.DataFrame({"cycle_index": [1], "ambient_temperature": [24.0]}),
            ["cycle_index", "ambient_temperature", "d_time_duration"],
        )

        self.assertEqual(missing, ["d_time_duration"])

    def test_soh_prediction_and_derived_rul_on_b0007_example(self) -> None:
        from src import inference

        example = pd.read_csv(config.EXAMPLE_INFERENCE_CSV_PATH)
        result = inference.predict_soh_from_frame(
            example,
            model_id="nasa_classic_4_randomforest",
            threshold=0.80,
        )

        self.assertIn("pred_soh", result.predictions.columns)
        self.assertIn("abs_soh_error", result.predictions.columns)
        self.assertGreater(len(result.predictions), 20)
        self.assertEqual(result.model_id, "nasa_classic_4_randomforest")
        self.assertEqual(result.battery_id, "B0007")
        self.assertEqual(result.threshold, 0.80)
        self.assertIsNotNone(result.eol_cycle)
        self.assertIsNotNone(result.remaining_cycles_at_first_cycle)
        self.assertGreaterEqual(result.remaining_cycles_at_first_cycle or 0, 0)


if __name__ == "__main__":
    unittest.main()
