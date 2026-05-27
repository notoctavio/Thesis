from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

from src import config


ARTIFACTS_AVAILABLE = config.ARTIFACTS_DIR.exists()


@unittest.skipUnless(ARTIFACTS_AVAILABLE, "Local artifacts directory is not available")
class StreamlitNarrativeTests(unittest.TestCase):
    def test_app_states_main_soh_direction_and_direct_rul_benchmark(self) -> None:
        app = AppTest.from_file("apps/streamlit_app.py")
        app.run(timeout=30)

        self.assertEqual(len(app.exception), 0)
        text = "\n".join(
            str(item.value)
            for group in [
                app.title,
                app.header,
                app.subheader,
                app.markdown,
                app.caption,
                app.info,
                app.warning,
            ]
            for item in group
        )

        expected_fragments = [
            "Abordarea principala a lucrarii este SOH-derived RUL",
            "Abordare principala: SOH -> prag EOL -> RUL derivat",
            "Benchmark comparativ: RUL direct",
            "De ce SOH-derived RUL?",
            "RMSE: penalizeaza erorile mari",
            "Aceasta zona este pentru verificari tehnice",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
