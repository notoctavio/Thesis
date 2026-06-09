# Data Directory

This directory contains the public, model-ready data required by the thesis
experiments, tests, and Streamlit demo.

## Tracked Structure

- `processed/battery_cycle_features_v2.csv`: final cycle-level feature table
  derived from the cleaned NASA battery dataset.
- `processed/*_feature_columns_v2.json`: feature column definitions used by
  the classical ML and SOH models.
- `splits/*.json`: train/validation/test battery split definitions and
  modeling scenarios.

## Local-Only Source Data

Raw and cleaned NASA source datasets are intentionally not tracked in git. If
the processed feature table must be regenerated, place the local source dataset
under a local-only folder such as `data/raw/cleaned_dataset/` and rerun the
notebooks or preprocessing workflow.
