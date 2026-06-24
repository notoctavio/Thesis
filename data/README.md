# Data Directory

This directory contains the public, model-ready data required by the thesis
experiments, tests, Streamlit demo, and CSV upload workflow.

## Tracked Structure

- `processed/battery_cycle_features_v2.csv`: final cycle-level feature table
  derived from the cleaned NASA battery dataset.
- `processed/*_feature_columns_v2.json`: feature column definitions used by
  the classical ML and SOH models.
- `splits/*.json`: train/validation/test battery split definitions and
  modeling scenarios.
- `examples/b0007_soh_inference_example.csv`: model-ready B0007 CSV used by
  the Streamlit `Predicție CSV` tab.

## Local-Only Source Data

Raw and cleaned NASA source datasets are intentionally not tracked in git. The
local source dataset used to regenerate the processed feature table should live
under `data/source/cleaned_nasa/`; then rerun the notebooks or preprocessing
workflow.

Download source dataset from Kaggle:

```bash
kaggle datasets download -d patrickfleith/nasa-battery-dataset -p data/source --unzip
```

If the extracted folder is named `cleaned_dataset`, rename it to
`data/source/cleaned_nasa/`.
