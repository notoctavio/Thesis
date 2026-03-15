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
    validate_datasets = os.getenv("VALIDATE_DATASETS", "1").lower() not in {"0", "false", "no"}

    required_files = [
        ROOT / "docs/ai/RULES.md",
        ROOT / "AGENTS.md",
        ROOT / "GEMINI.md",
        ROOT / "CLAUDE.md",
        ROOT / "CODEX.md",
        ROOT / "README.md",
    ]
    for file_path in required_files:
        all_ok &= check(file_path.exists(), "Required file exists", str(file_path.relative_to(ROOT)))

    canonical_ref = "docs/ai/RULES.md"
    for adapter in ["AGENTS.md", "GEMINI.md", "CLAUDE.md", "CODEX.md"]:
        adapter_path = ROOT / adapter
        content = adapter_path.read_text(encoding="utf-8") if adapter_path.exists() else ""
        all_ok &= check(canonical_ref in content, "Adapter references canonical rules", adapter)

    readme_text = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    all_ok &= check(
        "AI Assistant Documentation Convention" in readme_text,
        "README documents AI convention",
        "README.md",
    )

    if validate_datasets:
        raw_nasa = ROOT / "data_test/Raw Datasets/Nasa website datasets/5. Battery Data Set"
        raw_extracted = ROOT / "data_test/Raw Datasets/Datasets raw 5-56/battery_dataset"
        cleaned_root = ROOT / "data_test/Cleaned Datasets/Datasets 5-56 cleaned/cleaned_dataset"
        cleaned_data = cleaned_root / "data"
        metadata = cleaned_root / "metadata.csv"

        all_ok &= check(raw_nasa.exists(), "NASA zip directory exists", str(raw_nasa.relative_to(ROOT)))
        all_ok &= check(raw_extracted.exists(), "Extracted raw dataset directory exists", str(raw_extracted.relative_to(ROOT)))
        all_ok &= check(cleaned_root.exists(), "Cleaned dataset directory exists", str(cleaned_root.relative_to(ROOT)))
        all_ok &= check(cleaned_data.exists(), "Cleaned cycle CSV directory exists", str(cleaned_data.relative_to(ROOT)))
        all_ok &= check(metadata.exists(), "Metadata file exists", str(metadata.relative_to(ROOT)))

        zip_files = sorted(raw_nasa.glob("*.zip")) if raw_nasa.exists() else []
        extracted_dirs = sorted([p for p in raw_extracted.iterdir() if p.is_dir()]) if raw_extracted.exists() else []
        all_ok &= check(len(zip_files) >= 1, "NASA source zips found", str(len(zip_files)))
        all_ok &= check(len(extracted_dirs) >= 1, "Extracted raw groups found", str(len(extracted_dirs)))

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
        all_ok &= check(True, "Data-dependent checks skipped", "VALIDATE_DATASETS=0")

    print("\nRepository integrity:", "OK" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
