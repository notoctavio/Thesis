#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def check(condition: bool, title: str, details: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {title}: {details}")
    return condition


def main() -> int:
    all_ok = True
    validate_source_data = os.getenv(
        "VALIDATE_SOURCE_DATA",
        os.getenv("VALIDATE_DATASETS", "0"),
    ).lower() in {"1", "true", "yes"}

    required_files = [
        ROOT / "README.md",
        ROOT / "apps/streamlit_app.py",
        ROOT / "src/config.py",
        ROOT / "src/artifacts.py",
        ROOT / "src/inference.py",
        ROOT / "src/prediction.py",
        ROOT / "src/reporting.py",
        ROOT / "tests/test_artifacts.py",
        ROOT / "tests/test_inference.py",
        ROOT / "tests/test_prediction.py",
        ROOT / "tests/test_streamlit_narrative.py",
        ROOT / "requirements.txt",
        ROOT / "REPRODUCIBILITY.md",
        ROOT / "artifacts/README.md",
        ROOT / "data/README.md",
        ROOT / "data/examples/b0007_soh_inference_example.csv",
        ROOT / "data/processed/battery_cycle_features_v2.csv",
        ROOT / "data/processed/baseline_feature_columns_v2.json",
        ROOT / "data/processed/soh_feature_columns_v2.json",
        ROOT / "data/splits/modeling_scenarios_v1.json",
        ROOT / "artifacts/models/model_manifest.json",
        ROOT / "artifacts/metrics/all_model_test_comparison.csv",
        ROOT / "artifacts/metrics/soh_all_model_test_comparison.csv",
        ROOT / "artifacts/predictions/baseline_test_predictions.csv",
        ROOT / "artifacts/predictions/sequence_test_predictions.csv",
        ROOT / "artifacts/predictions/soh_all_test_predictions.csv",
    ]
    for file_path in required_files:
        all_ok &= check(file_path.exists(), "Required file exists", str(file_path.relative_to(ROOT)))

    readme_text = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    all_ok &= check(
        "Final Project Direction" in readme_text
        and "Application" in readme_text
        and "Reproducibility" in readme_text,
        "README documents project direction and app",
        "README.md",
    )

    manifest_path = ROOT / "artifacts/models/model_manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_entries = manifest.get("models", {})
        all_ok &= check(
            len(model_entries) >= 3,
            "Public model manifest has expected entries",
            str(sorted(model_entries.keys())),
        )
        for model_id, model_info in model_entries.items():
            model_path = ROOT / model_info.get("path", "")
            all_ok &= check(
                model_path.exists(),
                "Public model file exists",
                f"{model_id}: {model_path.relative_to(ROOT) if model_path.exists() else model_path}",
            )

    if validate_source_data:
        cleaned_root = Path(
            os.getenv(
                "SOURCE_DATA_ROOT",
                str(ROOT / "data/source/cleaned_nasa"),
            )
        )
        cleaned_data = cleaned_root / "data"
        metadata = cleaned_root / "metadata.csv"

        all_ok &= check(cleaned_root.exists(), "Cleaned source dataset directory exists", str(cleaned_root))
        all_ok &= check(cleaned_data.exists(), "Cleaned cycle CSV directory exists", str(cleaned_data))
        all_ok &= check(metadata.exists(), "Metadata file exists", str(metadata))

        csv_files = sorted(cleaned_data.glob("*.csv")) if cleaned_data.exists() else []
        metadata_rows = []
        if metadata.exists():
            with metadata.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                metadata_rows = list(reader)

            required_cols = {
                "type",
                "start_time",
                "ambient_temperature",
                "battery_id",
                "test_id",
                "uid",
                "filename",
                "Capacity",
                "Re",
                "Rct",
            }
            all_ok &= check(
                required_cols.issubset(set(reader.fieldnames or [])),
                "Metadata schema columns present",
                ",".join(sorted(required_cols)),
            )

        all_ok &= check(len(metadata_rows) > 0, "Metadata rows present", str(len(metadata_rows)))
        all_ok &= check(len(csv_files) > 0, "Cleaned cycle CSV files present", str(len(csv_files)))
        all_ok &= check(
            len(metadata_rows) == len(csv_files),
            "Metadata row count matches cleaned CSV count",
            f"{len(metadata_rows)} vs {len(csv_files)}",
        )

        if metadata_rows and csv_files:
            csv_names = {p.name for p in csv_files}
            metadata_names = {row.get("filename", "") for row in metadata_rows}
            all_ok &= check(
                metadata_names.issubset(csv_names),
                "All metadata filenames exist in cleaned data",
                f"missing={len(metadata_names - csv_names)}",
            )
    else:
        all_ok &= check(True, "Source data checks skipped", "set VALIDATE_SOURCE_DATA=1 to enable")

    print("\nRepository integrity:", "OK" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
