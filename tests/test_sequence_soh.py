from __future__ import annotations

import unittest

import pandas as pd

from scripts.train_sequence_soh import add_lag_features, build_sequences


class SequenceSohPreparationTests(unittest.TestCase):
    def test_add_lag_features_uses_only_previous_soh_for_residual_target(self) -> None:
        frame = pd.DataFrame(
            {
                "battery_id": ["B1", "B1", "B1"],
                "cycle_index": [1, 2, 3],
                "soh": [1.00, 0.96, 0.93],
            }
        )

        result = add_lag_features(frame)

        self.assertEqual(result["prev_soh"].round(4).tolist(), [1.00, 1.00, 0.96])
        self.assertEqual(result["soh_delta_target"].round(4).tolist(), [0.00, -0.04, -0.03])

    def test_build_sequences_pads_to_requested_length_and_keeps_prev_soh_meta(self) -> None:
        frame = pd.DataFrame(
            {
                "battery_id": ["B1", "B1"],
                "cycle_index": [1, 2],
                "start_time": ["2020-01-01", "2020-01-02"],
                "feature": [10.0, 20.0],
                "soh_delta_target": [0.0, -0.05],
                "prev_soh": [1.0, 1.0],
                "soh": [1.0, 0.95],
                "capacity_ah_clean": [2.0, 1.9],
                "rul_cycles": [1.0, 0.0],
            }
        )

        x, y, meta = build_sequences(
            frame,
            feature_cols=["feature"],
            target_col="soh_delta_target",
            seq_len=3,
        )

        self.assertEqual(x.shape, (2, 3, 1))
        self.assertAlmostEqual(float(y[0]), 0.0, places=4)
        self.assertAlmostEqual(float(y[1]), -0.05, places=4)
        self.assertEqual(x[0, :, 0].tolist(), [0.0, 0.0, 10.0])
        self.assertEqual(x[1, :, 0].tolist(), [0.0, 10.0, 20.0])
        self.assertIn("prev_soh", meta.columns)
        self.assertEqual(meta["prev_soh"].tolist(), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
